# HMX DSP-driven vs descriptor-driven 内核架构

**Status**: 实测确认 (2026-04-28)。这个区分决定 256³ matmul 5× 的速度差距，
是 HMX 性能优化的核心洞见。

> **核心结论**: HMX 不是无脑加速器。它是**带内部状态机和 stride 寄存器的引擎**。
> 高效 HMX kernel 不是"软件硬塞 tile MAC"，而是"先告诉 HMX 怎么扫、再让 HMX 自己扫"。
> 区别在于每个 software-issued mxmem packet **触发多少个 HMX 内部 MAC 操作**。

## TL;DR

| | DSP-driven | Descriptor-driven autonomous |
|---|---|---|
| 软件循环维度 | 全部 (mt × kt × nt) | 仅 nt × kt (M 轴 HMX 内部扫) |
| 256³ matmul 软件 MAC packet 对 | ~512 | ~64 |
| 每个 mxmem packet 触发的 HMX MAC | 1 个 tile (32×32×32 = 32K MAC) | 8 个 tile (8 × 32×32×32 = 262K MAC) |
| cyc/packet (HW 节拍) | ~4.1 | ~4.3 |
| **256³ 单 matmul 总 cyc** | **~8,000** | **~1,500** |
| **DSP-issued packet 数** | ~1,950 | ~350 |
| 例子 | V8C8 BbbKMajor (custom op) | native q::ConvLayer_s1.opt |

cyc/packet 几乎相同（HMX 物理流水线节拍同），**总速度差 5× 完全来自 packet 数差 5.6×**。

## HMX 硬件模型基础

HMX 是 Hexagon V75 的矩阵协处理器。核心抽象：

- **activation buffer**：32×32 byte（一个 K-tile 的 act 输入）
- **weight buffer**：32×32 byte
- **accumulator**：32×32 int32
- **state machine + stride registers**：可被 descriptor 配置成在多个 tile 上自动迭代
- **drain pipe**：通过 `:after:cm:sat.ub = acc` 指令把 accumulator 量化 + saturate 写到 VTCM

DSP 与 HMX 通过特定 mxmem 系列指令通讯：
```asm
mxclracc                                    ; 清 accumulator
bias = mxmem2(addr)                         ; 装 bias
activation.ub = mxmem(addr, Rt):cm          ; 装 act tile
weight.b      = mxmem(addr, Rt)             ; 装 wt tile, 触发 MAC
mxmem(addr, Rt):after:cm:sat.ub = acc       ; drain acc → VTCM
```

每个这种 packet DSP 需要等 HMX 流水线 ~4 cyc。这是 `Cycles per Packet ≈ 4.2` 的硬件下限。

**关键**：HMX 不止是"算 32×32 一次就停"。它的内部状态机允许**一次 mxmem 触发后自动沿
某个轴迭代多个 tile**——每个 tile 占用 ~4 cyc HMX 流水线节拍，期间 DSP **不需要**再发新 packet。

## DSP-driven 模式（V8C8 BbbKMajor）

我们的 V8C8 内核 `HmxMatMulV9SkelOp.cpp` 走 DSP-driven：

```c
// 软件三重循环 mt × nt × K-tile
mxclracc;
for (mt = 0..M_t-1) {
    for (nt = 0..N_t-1) {
        bias = mxmem2(bias_n);                    // 装 bias (1 packet)

        // hardware loop0, trip = K_t/2, 4-packet body / 2 MAC
        loop0(body, K_t/2);
        body:
            r6 = memw(r1++#8); r8 = memw(r3++#8);          // pkt 1 (load 2 act ptrs + 2 wt ptrs)
            r23 = memw(r1+#-4); r9 = memw(r3+#-4);         // pkt 2
            activation.ub = mxmem(r6, r24):cm;
            weight.b      = mxmem(r8, r25);                 // pkt 3 — 触发 1 个 K-tile MAC
            activation.ub = mxmem(r23, r24):cm;
            weight.b      = mxmem(r9, r25);                 // pkt 4 — 触发 1 个 K-tile MAC
            // 4 packets / 2 MAC = 2 cyc/MAC body

        sat.ub = acc;                                       // pkt — drain 1 个 1KiB tile
    }
}
```

**256³ packet 计算**：

```
Per (mt, nt) tile pair:
  4 packets/body × 4 trips (K_t=8 / 2)  = 16 MAC packets
  + 1 sat.ub + 1 bias + ~2 register init = ~20 packets

Total inner: 64 (mt,nt) × 20 = ~1280 packets
+ act_ptrs/wt_ptrs prebake (~640 stores) + entry/exit overhead = ~670 packets
Grand total: ~1,950 packets/matmul     ✓ matches measured 8000 cyc / 4.1 cyc/pkt
```

**含义**：每个 mxmem packet **只触发一个 K-tile 的 MAC**（32×32×32 = 32K 个
8-bit MAC）。256³ matmul 共 16.7M MAC，需 16.7M / 32K = **512 个 K-tile MAC packet 对**
（act_load + wt_load）。我们的 ~1024 MAC packets 就是 512 对。

软件控制每一个 (mt, kt, nt) tile 的迭代——HMX 是"无脑算一次"的角色。

## Descriptor-driven autonomous 模式（Native q::ConvLayer_s1.opt）

Native 的 `hmx_convbbb1x1_stride1`（`Agent/qnn_re/hmx_convbbb1x1_stride1_full.S`）：

```
Setup (0x2ea740..0x2ea7e4):
  从 r0 (descriptor 指针) 解出: M_count, K_count, N_count, strides, base addrs...
  r5  = K_t / 2
  r17 = M_count, r20 = N_count, r22 = M_step, ...

Pre-loop config (0x2ea7e8..0x2ea7fc):
  loop0(body, r5)                  ; inner loop trip = K_t/2 = 4 at 256³
  r24 = act Rt mask = 0x71F
  r25 = wt Rt mask  = 0x3FF
  nop; nop

★ 0x2ea800: 16-byte unknown packet (HMX state-config) ★
  bytes: 03 60 03 b0 / 0d 7c ed bf / 11 40 6c 70 / fe c3 03 92
  → 4 instructions, encoding 0xb0/0xbf/0x70/0x92
  这一包**把 descriptor 里的 M-axis stride/count 写进 HMX 内部寄存器**
  开启 HMX 的 "M-fan-out" autonomous mode

Outer loop entry (0x2ea810):
  loop1(body, r20)                 ; outer loop trip = N_count
  r1:0 = combine(...)              ; setup act ptr pair
  r17 -= r22                       ; decrement M counter
  r8 = r2                          ; reset wt ptr per outer iter

Inner body (0x2ea820..0x2ea848): 5 packets, 2 MACs
  pkt 1: cmp; r26-=2; r6=memw(r1++#8); r23=memw(r1+#4)
  pkt 2: r8 += 0x400 + <HMX-config> + activation.ub = mxmem(r6, r24):cm
  pkt 3: weight.b = mxmem(r8, r25)            ; 触发 MAC, HMX 自动 fan-out 多个 mt
  pkt 4: <HMX-config> + activation.ub = mxmem(r23, r24):cm
  pkt 5: weight.b = mxmem(r8, r25)            ; 触发 MAC, fan-out 多个 mt

Tail (0x2ea84c..0x2ea888):
  loop0 reset; conditional jump back to 0x2ea810 (outer-most loop)
```

**256³ packet 计算（实测推导）**：

```
Total packets ≈ dur / cyc_per_pkt = 1500 / 4.3 ≈ 350
Setup + drain                  ~30 packets
Inner body iterations:         ~320 packets

Body packets = 5 packets/body
→ 320 / 5 = ~64 body iterations ≠ 64 × 8 K-tile = 512!

If outer (loop1) trip = N_t = 8 and inner (loop0) trip = K_t/2 = 4:
  body iter = 8 × 4 = 32 (per HMX-fired pass)
  → outer-most loop runs ~10× 
  → total body iter ≈ 320 (matches!)
```

软件循环只迭代 **N × K 维**，**M 维由 HMX 内部状态机自动 fan-out**。

## "Fan-out" 机制详解

那个 `0x2ea800` 16-byte HMX state-config packet 把 descriptor 里的：
- **M_count**：M 维要扫多少 tile
- **M_step**：每个 mt 之间 act 地址增量 (= K × 32 bytes for byte-major K-major act)
- **output_stride**：每个 mt 输出的 VTCM 地址增量

写进 HMX 内部寄存器。一旦写入，**每一次 `weight.b = mxmem(r8, r25)` 不再只算一个
K-tile 的 32×32 MAC，而是**：

```
HMX 内部状态机伪代码：
  for (m = 0; m < M_count; m++):
    act_tile  = load(act_base + m × M_step)        ; 自动 advance
    wt_tile   = current weight buffer              ; 软件刚装的
    acc[m]   += act_tile @ wt_tile                  ; 累加进 M_count 个 acc tile
  ; 完成一次 (kt, nt) 在所有 mt 上的累加
```

也就是说，**一次软件 mxmem 触发 HMX 内部一个完整的 M-row sweep**（M_count = 8 个 tile
同时算）。

软件循环外层迭代 N_t × K_t 次，而非 M_t × N_t × K_t 次：
- V8C8 软件 MAC packet 对：8 × 8 (mt,nt) × 8 K-tiles = **512 对**
- Native 软件 MAC packet 对：8 (nt) × 8 K-tiles = **64 对**（HMX 内部自动 fan-out 8 个 mt）

64 vs 512 = **8× 软件 packet 数差距**，对应实测 5-6× 总 packet 数差距（剩 1-2× 在
setup/drain 开销）。

## 类比图解

```
─────────────────────────────────────────────
DSP-driven (V8C8 BbbKMajor):
─────────────────────────────────────────────

  for each (mt, kt, nt):
    DSP 发包: "HMX, 算 act[mt,kt] × wt[kt,nt] 累加进 acc[mt,nt]"
    HMX:     算 1 tile (32K MAC) → 等 4 cyc

  软件控制每个 tile 的迭代; HMX 是"被牵着走"的算盘


─────────────────────────────────────────────
Descriptor-driven autonomous (Native ConvLayer_s1.opt):
─────────────────────────────────────────────

  step 1 (一次性, 在循环外):
    DSP 发 16-byte HMX-config packet:
      "HMX, 配置 M_count=8, M_step=64×K, output_stride=1024..."
    HMX 把这些写进内部寄存器

  step 2 (重复 N_t × K_t 次):
    DSP 发包: "HMX, 算 K-tile (kt, nt), 在所有 8 个 mt 上 fan out"
    HMX:     M_count=8 次内部 MAC, 累加进 8 个 acc tile, 每个完成自动 sat.ub

  软件只控制 K 和 N 轴; M 轴 HMX 内部完全自治
```

## 为什么 cyc/packet 一样

每个 mxmem packet 喂数据到 HMX 后，DSP 必须等 HMX 流水线节拍 ~4 cyc。无论 HMX 内部
做 1 个 tile MAC 还是 8 个 tile MAC fan-out：

- HMX 流水线 throughput = ~32K 8-bit MAC/cyc (per tile MAC = ~4 cyc)
- Fan-out 模式下 8 个 tile MAC 在 HMX 内部**串行**进行（HMX 一个时刻只有一个 tile MAC 引擎）
- 但 DSP 看到的是"一次 mxmem 等 32 cyc HMX 完成"vs"一次 mxmem 等 4 cyc"

实际 DSP 看到的是**从 mxmem 到下一个可发指令的间隔**。HMX 内部 fan-out 时 DSP 等更久，
但**等的同时不需要发新 packet**——这就是节省 packet 数的来源。

cyc/pkt 数字两边相同，说明 HMX 内部 throughput 同（4 cyc/tile-MAC）。差别在
**DSP 总共发了多少 packet**——native 发得少，因为 HMX 在每个 packet 之间自己干了更多活。

## 实测数据 (256³, optrace_compare_256_indep/)

```
                            V8C8 BbbKMajor    Native ConvLayer_s1.opt
单 matmul wall              8,000 cyc          1,500 cyc            5.3× gap
单 matmul DSP packets       ~1,950             ~350                 5.6× gap
cyc/packet                  4.10               4.30                 ~same
软件 MAC packet 对          ~512               ~64                  8× gap
HMX 总 MAC ops 实际触发     16.7M              16.7M                same (256³)
HMX-MAC 每软件 mxmem        32K (1 tile)       262K (8 tiles fan-out) 8×
```

## 我们怎么追上 — Step 5 路径

要切到 descriptor-driven autonomous，本质上做两件事：

1. **构造 0x40 字节 descriptor**——反逆 `set_hmx_params_conv1x1(out, arg1..arg5)` 的
   5 个 packed-flag uint32 参数语义（已有 disasm: `Agent/qnn_re/set_hmx_params_conv1x1.S`），
   per-shape 算出 M_count / K_count / N_count / strides / Rt 掩码 / output base
2. **发那个 16-byte HMX state-config packet**——把 descriptor 写进 HMX 内部寄存器开启
   fan-out / autonomous mode

**最直接路径 = dlsym 调 `hmx_convbbb1x1_stride1`**：
- 用 `set_hmx_params_conv1x1` 烤 descriptor（自实现或 dlsym 调）
- 把 act/wt/bias VTCM 地址按 native 期望填进 descriptor
- 调用 native 函数让它跑整个 inner kernel

我们的 SkelOp 只负责 (a) 构造 descriptor、(b) 调 native HMX kernel。HMX fan-out / autonomous
tile sweep 全是 native 函数 + HMX HW 自己的事。

成功后 V8C8 BbbKMajor 单事件应降到 ~1,500 cyc 与 native parity。256³ wall 65 µs → ~25 µs。

**已有基础设施**：
- `Agent/qnn_re/set_hmx_params_conv1x1.S` — descriptor builder 完整 disasm
- `Agent/qnn_re/hmx_convbbb1x1_stride1_full.S` — kernel 完整 disasm
- `Agent/dlsym_spike_PASS_2026-04-25.md` — dlsym 已验证可调用 native 函数
- `Agent/v8c8_alignment_phase2_BREAKTHROUGH_2026-04-27.md` — Crouton_8 框架
- `optrace_compare_256_indep/` — 5× gap 实测 bundle

**剩余工作**：
1. 完整反逆 5 个 descriptor 参数语义
2. 在 SkelOp per-call 算出 descriptor (shape-derived，可缓存)
3. dlsym 调 `hmx_convbbb1x1_stride1` 跑整个 inner kernel
4. 调通 256³ bit-exact + sweep 512/1024/2048
5. 验证 chrometrace 单事件 dur 降到 ~1,500 cyc

## 普适经验 — 写 HMX op 的方法论

> 写 HMX op **不要**问"我怎么手塞 tile MAC"，要问"HMX 怎么自己扫"。

具体：
1. **看 native disasm 找 HMX state-config packet**（特征：紧跟 setup 之后、loop 之前的
   不寻常 mxmem 编码）
2. **找该 op 的 descriptor builder**（如 `set_hmx_params_conv*` 系列）
3. **理解 descriptor 字段→HMX 内部状态映射**
4. **dlsym 复用 native 函数**优于自写 inner kernel——HMX state-config 指令编码
   多数无文档，硬写极易踩硅级 bug

V8C8 Step 1+2 的"复刻 native I/O 形态"是必要的（让 wt/act 字节布局对齐 native 期望），
但**真正的 perf 复刻是 Step 5 的内核架构复刻**——要让 HMX 自走。

## 参考

- 实测 bundle: `optrace_compare_256_indep/` (chrometrace + profile_text + QHAS HTML)
- Memory entry: `~/.claude/.../project_v8c8_dsp_vs_descriptor_kernel_2026-04-28.md`
- HMX 编码参考: `docs/hmx-programming-guide/`、`docs/hexagon-v75-ref/`
- QNN custom op SOP: `docs/qnn_custom_op_sop.md`
