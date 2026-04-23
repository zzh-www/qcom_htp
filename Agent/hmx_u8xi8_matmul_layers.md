# HMX `u8 × i8` 矩阵乘法的全栈分层实现

**目标**：以 v75 HMX 上 32×32×32 `u8 × i8 → i32` 矩阵乘法为主线，从
最上层（C 代码/ISA）到最下层（cell 里的乘法器电路）把每一层都讲透。

**证据标注**：
- **(F)** 由 ISA 头、模拟器反汇编、设备二进制、本仓库实测直接支撑。
- **(P)** 从 (F) 推断出的最可能架构，给出推理链，可被 RTL 或 silicon 探针证伪。

**参考源**：
- `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hmx_hexagon_protos.h` — ISA 头
- `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/lib/iss/libhexagonissv75.so` — v75 模拟器（x86, 带调试符号）
- `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` — 设备二进制
- `example/hmx_matmul_int16/int16_matmul_hmx.c` — 本仓库端到端 bit-exact 的 HMX MAC 参考实现
- `example/hexagon_hmx_matmul_native_int.md` — 本仓库 tile 布局实测日志

---

## 目录

```
L0 要计算什么（数学定义）
L1 程序员视角：一段 inline-asm
L2 ISA 层：VLIW packet 与指令编码
L3 内存层：VTCM 中的 tile 字节布局
L4 发射层：一条 mxmem 指令的 "credit-driven" 展开
L5 阵列层：32×32 MAC cell 的几何与连线
L6 Cell 层：单个 MAC cell 做一次 u8×i8
L7 累加器层：int32 accumulator 与 swap
L8 读回层：bias + convert + saturate
端到端 trace：一个输出 cell 的全路径
附录 A：改成 int4/int2/int1/fp16 每一层怎么变
附录 B：为什么 activation tile 是 2 KiB 而 weight 只有 1 KiB
```

---

## L0 要计算什么 (F)

```
A ∈ u8^{32×32}     (activation, zero-point 已抵消)
W ∈ i8^{32×32}     (weight, signed)
C ∈ i32^{32×32}    (output accumulator, 量化前)

C[i, j] = Σ_{k=0}^{31} A[i, k] · W[k, j]
```

累加器范围：
- 每 K 步 `|A · W|` ≤ 255 × 127 = 32 385
- K = 32 累加 ≤ 1 036 320 ≈ 2^20（适合 int32 容纳；但后面会看到 HMX 只能
  一次读回 16 bit，所以需要 dual-scale readback 才能无损读出）

---

## L1 程序员视角：一段 inline-asm (F)

本仓库 `int16_matmul_hmx.c:44-52` 实际跑通的代码：

```c
static inline void hmx_load_pair_u8_i8(const void *act, const void *wt) {
    asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(2047),
           "r"(wt),  "r"(2047)
        : "memory");
}
```

配合以下序列完成一个 32³ MAC + readback：

```c
hmx_load_bias_i(bias_lo);            // bias = mxmem(Rs)      — 载入 128×u16 scale
hmx_clracc_i();                       // mxclracc              — 清零 int acc
hmx_load_pair_u8_i8(act_tile, wt_tile); // 上面那个 VLIW packet — 1 次 MAC 覆盖全 K
hmx_store_acc_uh_2x1_retain(out_lo); // mxmem:after:retain.uh = acc:2x1
hmx_load_bias_i(bias_hi);            // 换 scale
hmx_store_acc_uh_2x1(out_hi);        // mxmem:after.uh        = acc:2x1
```

两次 convert 读同一个 acc，用不同 scale 重建完整 int32（见 L8）。

**从程序员视角，一次"32×32×32 MAC"就是一个 VLIW packet**——这个抽象下面
所有细节都被硬件和 ISA 隐藏。下面逐层剥开。

---

## L2 ISA 层：VLIW packet 与指令编码 (F)

### 两条 slot-0 指令同 packet 发射

VLIW packet 花括号内的两条指令会**同一个 PC 点**发射出去：

```
{ activation.ub = mxmem(Rs1, Rt1)       ; I1 — 载入 act tile 句柄
  weight.b      = mxmem(Rs2, Rt2) }     ; I2 — 载入 wt  tile 句柄
```

按 `hmx_hexagon_protos.h` 的规定，这两条都**执行在 SLOT 0**。Hexagon 的
packet 只有 4 个 slot，通常一个 slot 只能放一条 SLOT0 指令。但 HMX 的
activation.* 和 weight.* 是**伴生对**（paired operands），硬件允许它们
共占一个 packet——这是 HMX 专门开的规则。

### 32-bit 指令编码（从设备二进制抠出来的真实字节）

```
activation.ub = mxmem(r0, r6):above     0x92 00 46 ee
weight.b      = mxmem(r23, r9):dilate   0x92 17 e9 eb
```

字段分解（详见 `hmx_int4_combos_analysis.md` 附录 A 的指令编码统计）：
```
bits [31:28]  ICLASS    = 0x9   （HMX 指令族，和 HVX/LD/ALU 区分开）
bits [27:24]  op24      = 0x2   （"MAC-load" 子族，activation 和 weight 共用）
bits [25:16]  mid16     = 0x06 (act)  / 0x08 (wt-base)  ← 区分 act 还是 wt
bits [15:8]   source reg Rs  (载体指针)
bits [7:4]    modifier bits  (:above / :dilate / :single / :deep / :cm)
bits [3:0]    type code:
              0x0 = b (int8)    0x1 = n (int4)    0x2 = c (int2)
              0x3..0xe 预留      0xf = hf (fp16)
```

关键观察：**`activation.ub` + `weight.b` 在 ISA 编码层面是同一条 MAC-load
指令的两种参数化**，共享 opcode 族，只在低 4 位 type code 上不同。意味着
硬件里只有一套解码器在处理所有 HMX MAC-load 变种。

### 这条 packet 要做什么

执行语义：
1. `activation.ub = mxmem(Rs_act, Rt_act)`
   将 `Rs_act` 指向的 2 KiB VTCM 区间 **排队到 HMX 的 activation FIFO**。
   `Rt_act = 2047` 是一个 encode 了"读满整个 2 KiB"的控制字。
2. `weight.b = mxmem(Rs_wt, Rt_wt)`
   类似地排队 1 KiB 的 weight tile 到 HMX 的 weight FIFO。
3. 两个 FIFO 同时就位 + 累加器就位后，HMX **自启动一次 K=32 的 MAC**，
   写入 1024 个 int32 accumulator cell。
4. 指令本身**立即退休**，Hexagon 核可以继续发射后续 packet；HMX 在后台
   消费 FIFO（credit driven，见 L4）。

---

## L3 内存层：VTCM 中的 tile 字节布局 (F)

（实测见 `example/hexagon_hmx_matmul_native_int.md` 的 single-hot-byte
probes。）

### 激活 tile (2 KiB)

```
A_byte(phys_row, K, stream) =  128 · phys_row + 4 · K + (stream ? 3 : 1)
    phys_row ∈ 0..15
    K        ∈ 0..31
    stream   ∈ {0, 1}
```

- 字节位置 `4·K + 0` 和 `4·K + 2` **必须置零**（硬件会读但 int 模式下
  不用；这些 slot 留给 fp16 activation 的高字节）。
- 逻辑行 `ir ∈ 0..31` → `phys_row = ir & 15, stream = ir >> 4`。
  - stream 0 服务 logical rows 0..15
  - stream 1 服务 logical rows 16..31

可视化（128 B 一行 = 一条 HVX 向量）：

```
phys_row=0 的 128 B：
[_ a(0,0) _ a(16,0)] [_ a(0,1) _ a(16,1)] ... [_ a(0,31) _ a(16,31)]
  ^ "_" = 0 bytes（忽略槽）
  ^ stream=0=a[ir=0] 占 byte 1，stream=1=a[ir=16] 占 byte 3
```

一个 HVX 向量就装下了"logical 行 0 和 logical 行 16"的全部 32 个 K 元素。

### 权重 tile (1 KiB)

```
W_byte(K, col) =  128 · (K >> 2) + 4 · col + (K & 3)
    K   ∈ 0..31
    col ∈ 0..31
```

可视化：
```
线 0 (128 B, 覆盖 K=0..3 的所有 32 列)：
[w(0,0) w(1,0) w(2,0) w(3,0)] [w(0,1) w(1,1) w(2,1) w(3,1)] ... [w(0,31) w(1,31) w(2,31) w(3,31)]

线 1 (128 B, 覆盖 K=4..7 的所有 32 列)：
...

线 7 (128 B, 覆盖 K=28..31 的所有 32 列)：
```

一个 HVX 线装 "4 个连续 K 行 × 32 列"。共 8 条线覆盖 K = 32。

### Bias tile (256 B)

128 × u16 per-column scale，每 u16 一个 f16 值。
- `0x4000` = f16(2.0) → 有效 scale = 1.0（"identity"）
- `0x2000` = f16(2^-7) → 有效 scale = 2^-8（HMX 内部有 /2）

Bias 是按输出列索引的，32 个逻辑列 × 2 streams × 2（retain 的 slot 0/1）
= 128 个 u16。

---

## L4 发射层：一条 mxmem 指令的 credit-driven 展开 (F + P)

从 `hmx_int4_combos_analysis.md` 附录 A 反汇编的 `hmx_convhnh_5x5_stride1`（设备 0x213e60）可以看到：在内层循环里**只发一
次 MAC packet 就覆盖全 K**——Hexagon 核发射后不再等 HMX 执行结束，而是立
刻跑下一个 packet。

这是怎么做到的？(F) 模拟器里有这些证据：
- `arch_hmx_set_credits(n)` — 设置 HMX 可挂起的未完成事务数
- `compute_derived_stat_HMX_CYCLES_RUNNING_PERCENT` — HMX 有自己的运行计数器
- `compute_derived_stat_HMX_STALLS` — HMX 可 stall 核心
- `compute_derived_stat_HMX_SCALAR_OVERHEAD` — 核和 HMX 间有缝隙

(P) 从这些信号推断出的执行模型：

```
Hexagon core (scalar/HVX)    HMX co-processor
─────────────────────────────────────────────
    ↓ issue packet
{ act.ub = mxmem(... 2047)      act FIFO:  [ptr, len=2KiB]   ← enqueue
  wt.b  = mxmem(... 2047) }     wt  FIFO:  [ptr, len=1KiB]   ← enqueue
    ↓ packet retires
    (core 继续跑)              ┌─────────────────┐
                               │ HMX internal    │
                               │ loop (8 cycles) │
                               │   cycle 0: read act line 0, wt line 0, MAC
                               │   cycle 1: read act line 1, wt line 1, MAC
                               │   ...
                               │   cycle 7: read act line 7, wt line 7, MAC
                               └─────────────────┘
```

关键点：
- **一条 mxmem 不是 1 个 MAC 周期，而是 "启动一个到 (2KiB/128B)=16 行 × 
  8 周期 的 HMX 任务"**。核侧看到的指令延迟是 1（立即退休），吞吐是
  大约 "一个 32×32×32 tile / 8 HMX 周期"。
- 如果核在 HMX 还没做完时又发了一条 mxmem，硬件通过 **credits** 控制最多
  可挂 K 个任务（见 `arch_hmx_set_credits`）。credit 耗尽时核 stall，
  对应 `HMX_STALLS` 计数器。
- `Rt = 2047` 的 11 位立即数编码"本次任务吃完整个 2 KiB 区间"；不同值
  可以编码"只吃前 N 字节"或"跳步"，对应 conv 的 stride/dilate 模式。
  这就是 `:above / :single / :deep / :dilate` 修饰符的来源。

### int8 MAC 的内部 8 周期（P）

既然权重 tile 是 8 × 128 B，每 cycle HMX 消费一条 128 B 线（= 4 个 K
× 32 列），那么：
- Cycle 0：act 线 0 + wt 线 0 → K=0..3 的 4 次 outer product，累加
- Cycle 1：act 线 1 + wt 线 1 → K=4..7
- ...
- Cycle 7：act 线 7 + wt 线 7 → K=28..31

等等——activation tile 是 2 KiB = 16 条线，而 weight tile 是 1 KiB = 8 条线。
activation 为什么要 2×？因为每条 act 线只覆盖 **1 个 phys_row** × 全部 K（见
L3）；需要 16 条线对应 16 个 phys_row。weight 线一条就覆盖 4 个 K × 全部
32 列。

实际上内部 cycle 计数是 **act 线数 = 16**？还是 **wt 线数 = 8**？从执行
时间看（用户 memory 里 fp16 32³ ≈ 216 ns / 700 MHz ≈ 150 cycles，远多于
16），应该和 tile 总字节数 + 其他开销有关。具体微架构周期数本文不展开——
**软件只需要知道"一条 mxmem = 一次完整的 K=32 MAC"**。

---

## L5 阵列层：32×32 MAC cell 的几何 (F + P)

从模拟器符号 `arch_get_hmx_spatial_size` 和 `arch_get_hmx_channel_depth`
可以推断 HMX 是二维阵列。从 tile 布局的几何反推：

- **空间维度**：32 列（输出列 j）× 32 行（输出行 i）= **1024 个 MAC cell**。(F, 与输出 tile 尺寸吻合)
- **每个 cell 内部有一个 int32 accumulator 和一个 fp accumulator**。(F, 由
  `mxclracc` vs `mxclracc.hf` 两条独立清零指令证明)
- **每 cycle 整个阵列从 act FIFO / wt FIFO 各消费 128 B**，分发到
  32 列，列内再按 K 扩展。

### 分发模式 (P)

一条 128 B weight 线有 "4 K-rows × 32 cols" 的 int8。分发到 32 列的 cell
时：
- 列 j 收到该线对应列 j 的 4 个 K 字节：`{ W[K,j], W[K+1,j], W[K+2,j], W[K+3,j] }`
- 在一个 HMX 内部周期里，**每 cell 连续做 4 次 MAC**（K=K0, K0+1, K0+2, K0+3）。
  或者更可能是：cell 内部**本来就是 4-way 展开的**，一周期吃 4 个 K 产物。

一条 128 B activation 线装两个 phys_row（stream=0 和 1）的各 32 个 K。
分发时：
- 行 i 收到该线对应 phys_row 的 4 个 K 字节。注意每个 cell 服务两个
  logical row（ir 和 ir+16，通过 stream 区分）。
- Output cell 其实是 **16 phys_row × 2 stream × 32 col = 1024 logical cells**。

### "单一 MAC 阵列" vs "fp/int 两套阵列"？(P)

`hmx_int4_combos_analysis.md` §3.1 的证据（两套 acc、两条 clracc、两种 PMU 统计）只说明**累加器
和控制逻辑分开**，不一定是两个物理 MAC 阵列。最节省硅面积的方案是：

> **同一 32×32 cell 阵列，每 cell 内部的乘法器单元可以重配置为定点 MAC 或
> 浮点 MAC。两套累加器物理分开但乘法器共享。`mxclracc` / `mxclracc.hf` 分
> 别重置各自的累加器组。**

验证或证伪这点需要 RTL，本文只提一种合理假设。

---

## L6 Cell 层：一次 u8 × i8 MAC 到底是怎么算的 (F 骨架 + P 细节)

### 功能语义 (F)

ISA 层承诺：
```
acc[i][j] += (uint8)a_byte × (int8)w_byte
```

模拟器里体现为 4 行 x86（`hmx_mult_fxp` @ 0x41b630）：
```asm
movsbl  %r9b, %edx     ; edx = sign_extend(w_byte as int8)
movzbl  %r8b, %esi     ; esi = zero_extend(a_byte as uint8)
imul    %edx, %esi     ; 32-bit signed multiply
add     %esi, ACC[cell] ; int32 accumulate
```

这是功能等价物（模拟器的实现），不是硅的 RTL。

### 物理乘法器：不是 u8×i8 基元，但输出也不是 N 路并联 (F+P，硅级验证过)

两个容易犯的极端都错：

- **错误 A**："硅里就是一个物理 u8×i8 乘法器，int4 通过叫 2 次复用来实现"。
  → 错在 cycle 数：如果 int4 是"多叫一次"，cyc/packet 应该是 int8 的 2 倍。
     **硅实测**`weight.n` 的 cyc/packet 只比 `weight.b` 慢 4%（测量噪声）。
- **错误 B**："cell 内部有 8 路独立输出，int4 一 packet 产 2 个不同 K-stride 的
  结果"。 → 错在输出值：若如此，每个 cell 输出 cell 只收到一个 nibble 的贡献。
  **硅实测**`byte=0x11`（两个 nibble=1）→ out=32 = 16×(1+1)，说明**两个
  nibble 都被累加到同一个输出 cell**。

与所有观测一致的模型（P）：

> **每 cell 一个 int32 累加器 + 一个单乘法器 + 一个 sub-byte 展开器**。
> load 阶段把一字节拆成 N 个 sub-byte（N = 1/2/4/8 对应 int8/int4/int2/int1），
> MAC 阶段沿 K 维**顺序**发射 N 次 u8 × sub-byte 的乘加，全部累加到同一个
> 输出 cell。

物理上乘法器的位宽可能是 8-bit 完整乘法器，也可能是 bit-serial 8 路合并。
都不影响观察到的功能/cycle 行为；区分需要 RTL。

### u8×i8 模式下 cell 内部一次 MAC 的可能电路 (P)

```
Inputs:   a = uint8,  w = int8

Option A — 完整 8×8 signed-unsigned 乘法器（Baugh-Wooley 或等价结构）：
  product_16 = (int16)w × (uint16)a
  acc_32    += sign_extend_32(product_16)

Option B — 8 路 bit-partial 然后求和：
  p[k]  = a & (bit_k(w) ? 0xFF : 0x00)    for k=0..7
  product = Σ_{k=0..6} (p[k] << k)  −  (p[7] << 7)   ; 两补符号处理
  acc_32 += sign_extend_32(product)
```

这两个在功能和 cycle 上无法区分。Option B 更契合 sub-byte 共享硬件——
把 `weight.n` 解释为"只用 4 个 partial 的 int4 乘法器"即可，其余 4 个
partial 留作下一个 K 步继续累加。

### sub-byte 路径对应的 cell 行为（硅验证）

- `weight.b`: 32 bytes 读进，每 byte 1 次 u8×i8 → 32 次 MAC / cell → 1 输出
- `weight.n`: 16 bytes 读进，每 byte 分 2 nibble → 32 次 u8×i4 → 1 输出
- `weight.c`:  8 bytes 读进，每 byte 分 4 crumb → 32 次 u8×i2 → 1 输出
- `weight.ubit`: 4 bytes 读进，每 byte 分 8 bit → 32 次 u8×i1 → 1 输出

**每 cell 每 packet 始终产 1 个 int32 输出 + 32 次 K-step 累加**。sub-byte
权重省的是"要读多少 byte"而不是"要做几次 MAC"。

### activation 路径

activation 始终是 uint8，无 sub-byte 展开（ISA 里也没 `activation.n`）。
fp16 activation 走独立的 `xfp` 通道（`hmx_mult_xfp`）。两套独立 accumulator。

### 证据等级总结

| 论断 | 级别 | 依据 |
|------|:----:|------|
| 每 cell 每 packet 1 个 int32 输出 + 32 次 K 累加 | **F** | 真机 `probe_subbyte_device.c` 实测输出规律 |
| sub-byte = 读更少 byte，非多路并联输出 | **F** | 硅上 `byte=0x11 → 32`，不是 2 个独立的 16 |
| cyc/MAC 与 weight 宽度无关（< 5%）| **F** | 硅实测 `HAP_perf_get_pcycles` |
| cell 内 8-partial adder tree 是 Option B | **P** | 是 sub-byte 共享硬件的最简单实现；RTL 未知 |
| fp16 物理独立阵列或复用乘法器 | **P** | 功能上独立，硅级无从区分 |

---

## L7 累加器层：int32 accumulator (F)

### 物理布局

- 1024 个 int32 cell，排成 32 × 32 的逻辑网格。
- 另有一套 fp accumulator（ xfp 格式，内部比 fp16 更宽以保留中间精度）。
- 整套 accumulator 被进程唯一持有（`qurt_hmx_lock`）。

### 状态切换

- `mxclracc`       → 清零 int32 acc。int 乘加模式被激活。
- `mxclracc.hf`    → 清零 xfp acc。fp 乘加模式被激活。
- `mxswapacc`      → 在两套 int32 acc 之间交换（支持 double-buffered conv）。
  设备代码 `hmx_convhnh_5x5_stride1` 用 `mxswapacc` 在 5×5 kernel 的前 2
  行和后 3 行间切换，实现 overlap-add。本仓库 matmul 用不到。

### 累加器容量与溢出

- 每 cell int32 → 范围 ±2^31。
- K=32 u8·i8 最大 ≈ 2^20，远低于 int32 上限，不会溢出。
- K=8192 u8·i8 最大 ≈ 2^29，仍安全。
- **但读回路径只能一次取 16 bit**，所以需要 L8 的 dual-scale 方案才能完整
  拿回 int32。

---

## L8 读回层：convert + saturate (F)

### `mxmem(Rs, 0):after.uh = acc:2x1`

执行顺序（以 convert-after-MAC 的 `:after` 为例）：

1. **Per-column bias 相乘**：取当前 bias tile 的 u16[j]（f16 格式），与
   acc[i][j] 相乘。HMX 内部还会再除 2（硬件常数）。
   ```
   intermediate[i][j] = acc[i][j] * f16_to_f32(bias[j]) / 2
   ```
2. **格式转换**：int32 intermediate → uint16。`:sat` 变体做 saturating
   clamp 到 [0, 65535]；无 `:sat` 是 wrap mod 2^16。
3. **写 VTCM**：1024 个 u16 = 2 KiB 写到 `Rs`。
4. **累加器处理**：
   - 默认：convert 完后累加器被消费掉（下次 MAC 必须先 `mxclracc`）。
   - `:retain` 变体：保留累加器不变（为下次 convert 用不同 scale 做准备）。

### 输出 tile 布局

```
out_u16[phys_row · 64 + 2 · col + stream]
```
每 phys_row 64 个 u16 = 2 cols × 2 streams × 32 = 128 字节。
共 16 phys_row × 128 B = 2 KiB。

逻辑映射：
```
out[ir][jc] = out_u16[(ir & 15) · 64 + 2 · jc + (ir >> 4)]
```

### Dual-scale readback 拿回 int32

单次 convert 的输出是 16 bit，而 int32 acc 有 32 bit。要无损读回：

```c
// Pass 1: scale = 1.0 (bias = 0x4000)
mxmem(OUT_LO, 0):after:retain.uh = acc:2x1
  → OUT_LO[j] = acc[i][j] mod 2^16        (低 16 位)

// Pass 2: scale = 2^-8 (bias = 0x2000)
mxmem(OUT_HI, 0):after.uh        = acc:2x1
  → OUT_HI[j] = floor(acc[i][j] / 256) mod 2^16  (高 16 位，带符号)

// CPU 侧 stitch:
acc_int32 = ((int16_t)OUT_HI[j] << 8) | (OUT_LO[j] & 0xFF)
```

本仓库 `int16_matmul_hmx.c:108-141` 已端到端 bit-exact 验证这个还原是
精确的（3 个场景 × 1024 cells × 完全匹配 int64 oracle）。

---

## 端到端 trace：一个输出 cell 的全路径 (F)

以计算 `C[5][17] = Σ_{k=0..31} A[5][k] · W[k][17]` 为例，跟着数据走一遍。

### CPU pack 阶段

**Activation**: 填 A[5][k] 到 act_tile。
- 逻辑行 ir = 5 → phys_row = 5 & 15 = 5, stream = 5 >> 4 = 0
- byte_offset(K) = 128·5 + 4·K + (0 ? 3 : 1) = 640 + 4·K + 1
- 例如 A[5][0] → act_tile[641]; A[5][1] → act_tile[645]; ...; A[5][31] → act_tile[765]
- 这些位置之间的 `4·K + 0` 和 `4·K + 2` 保持 0。

**Weight**: 填 W[k][17] 到 wt_tile。
- col = 17
- byte_offset(K) = 128·(K>>2) + 4·17 + (K&3) = 128·(K>>2) + 68 + (K&3)
- W[0][17] → wt_tile[68]; W[1][17] → wt_tile[69]; W[2][17] → wt_tile[70]; W[3][17] → wt_tile[71];
  W[4][17] → wt_tile[196]; ...; W[31][17] → wt_tile[959]

### MAC 发射

`mxclracc` → 阵列 acc[5][17] = 0。

`{ activation.ub = mxmem(&act_tile, 2047); weight.b = mxmem(&wt_tile, 2047) }`
发射后，HMX 内部（近似）：

```
for k_line in 0..7:              # 8 条 weight 线
    act_line  = act_tile[k_line*128 : k_line*128+128]
                # 或者更精确：activation 以 phys_row 索引，需要映射
    wt_line   = wt_tile [k_line*128 : k_line*128+128]
    
    # cell (phys_row=5, stream=0, col=17) 收到的输入：
    a_bytes = [ act_line reads at positions corresponding to (5, stream=0, K=4·k_line..4·k_line+3) ]
    w_bytes = [ wt_line  reads at positions corresponding to (K=4·k_line..4·k_line+3, col=17) ]
    
    # cell 内部 4 次 MAC（K-wise unroll）:
    for K in {4·k_line, 4·k_line+1, 4·k_line+2, 4·k_line+3}:
        acc[5][17] += (uint8) A[5][K] × (int8) W[K][17]
```

### Cell 内部（L6）

每次 MAC（以 K=0 为例，A[5][0] = 0x80, W[0][17] = 0x7F 为一个具体数）：

```
a = 0x80 = 1000_0000_b (uint8 = 128)
w = 0x7F = 0111_1111_b (int8 = +127)

Stage 1 - 8 个 partial (u8 AND w[k]):
  p[0] = 0x80 & FF = 0x80     (w[0]=1)
  p[1] = 0x80 & FF = 0x80     (w[1]=1)
  p[2] = 0x80 & FF = 0x80
  p[3] = 0x80 & FF = 0x80
  p[4] = 0x80 & FF = 0x80
  p[5] = 0x80 & FF = 0x80
  p[6] = 0x80 & FF = 0x80
  p[7] = 0x80 & 00 = 0x00     (w[7]=0, 符号位)

Stage 2 - 16-bit 加法树 (p[7] 用减号):
  product = 0x80 + (0x80<<1) + (0x80<<2) + ... + (0x80<<6) - (0x00<<7)
          = 128 × (1+2+4+8+16+32+64) − 0
          = 128 × 127 = 16256
          = 0x3F80

Stage 3:
  acc[5][17] += sign_extend_32(0x3F80) = +16256
```

32 次 MAC 累加完毕后，假设 acc[5][17] 总和 = 0x0023_45AB（随便一个例子）。

### 读回阶段 (L8, dual-scale)

Pass 1：bias[17] = 0x4000 (f16 2.0 → scale 1.0)
```
intermediate = 0x0023_45AB × 1.0 = 0x0023_45AB
OUT_LO[phys_row·64 + 2·17 + 0] = 0x45AB  (低 16 bit)
```

Pass 2：bias[17] = 0x2000 (f16 2^-7 → scale 2^-8)
```
intermediate = floor(0x0023_45AB / 256) = 0x0000_2345
OUT_HI[...] = 0x2345
```

CPU 侧重建：
```
acc = ((int16_t)0x2345 << 8) | (0x45AB & 0xFF)
    = 0x0023_4500 | 0x0000_00AB
    = 0x0023_45AB  ✓
```

---

## 附录 A：改成其他 weight 类型，每层怎么变

| 层 | int8 `weight.b` | int4 `weight.n` | int2 `weight.c` | int1 `weight.ubit` | fp16 `weight.hf` |
|----|-----------------|-----------------|-----------------|---------------------|-------------------|
| L1 ISA | `weight.b` | `weight.n` | `weight.c` | `weight.ubit` | `weight.hf` |
| L3 wt tile size | 1024 B | 512 B | 256 B | 128 B | 2048 B |
| L3 K-rows per 128B line | 4 | 8 | 16 | 32 | 2 |
| L6 unpacker | identity | `nibble_from_byte` | `crumb_from_byte` | `1bit_from_byte` | `unpack_none` |
| L6 MAC 基元 | `fxp` (u8×i8→i32) | `fxp_subbyte` | `fxp_subbyte` | `fxp_subbyte` | `xfp` |
| L6 每 byte 展开 sub-byte 数 | 1 | 2 | 4 | 8 | — |
| L6 每 cell 每 packet 输出 | **1 个 int32** | **1 个 int32** | **1 个 int32** | **1 个 int32** | 1 个 xfp |
| L6 cyc/packet (真机) | ~45.8 | ~47.5 (+4%) | ~47.5 (+4%) | ~47.6 (+4%) | ~same |
| L7 accumulator | int32 | int32 | int32 | int32 | xfp |
| L7 clear inst | `mxclracc` | `mxclracc` | `mxclracc` | `mxclracc` | `mxclracc.hf` |
| L8 readback | `:sat.uh` / dual-scale | 同 | 同 | 同 | `:after.hf` |

**sub-byte 优势的正确解读**：每 packet MAC 数和 cycle 数基本不变，但 weight
tile 字节数 1/N。在 VTCM-bound 场景（v75 MatMul 典型）下，tile 缩小直接
翻译成实际 speedup——marketing 的 "int4 = 2× int8" 来自这里。

**"int8 活动激活"和其他 activation 类型**：activation 只有 `.ub` / `.hf` / `.f8`
三种原生类型。int16 或 signed int8 activation 必须在 CPU 侧分解（见
`hmx_int4_combos_analysis.md` §5）。

---

## 附录 B：为什么 activation tile 是 2 KiB 而 weight 只有 1 KiB

两者都覆盖 32 × 32 的 int8 数据，按字面算应该都是 1 KiB。但 activation
tile 是 2 KiB（1024 byte 数据 + 1024 byte 的忽略槽）。原因：

> Activation tile 的 byte layout **要和 fp16 activation 兼容**。fp16 每
> 元素 2 byte，32 列 × 32 K 需要 2 KiB。硬件不想给 int 和 fp 两套不同
> 的 act-FIFO 几何，于是 int8 activation 也占 2 KiB，只用每个 16-bit
> slot 的高字节（位置 `4K+1` 和 `4K+3`）。

weight 侧则不共享：weight tile 对 int8 就是 1 KiB，对 fp16 是 2 KiB。从 
`hmx_wgt_init` 的符号共用也能看出——weight 加载路径按 type 参数化 tile
大小，没有"永远 2 KiB"的约束。

这就是为什么 `hmx_int4_combos_analysis.md` §4.2 里 "K_rows_per_line = 4 / element_bytes" 对 weight
适用（int8: 4, fp16: 2, int4: 8），但对 activation 不完全适用（int8 和
fp16 都是 2，因为 int8 在 16-bit slot 里只用一半）。

---

## 附录 C：一张图把所有层连起来

```
             用户代码 (L1)
               ↓ asm volatile { act.ub = mxmem; wt.b = mxmem }
             ISA (L2) 32-bit instruction encoded (ICLASS=9)
               ↓ hexagon-core SLOT0 decode
             VTCM (L3) 2KiB act tile + 1KiB wt tile
               ↓ mxmem queues tile pointer+len into HMX FIFO
             HMX dispatcher (L4) credits + multi-cycle task
               ↓ per cycle: read 128 B act line + 128 B wt line
             Cell 阵列 (L5) 32 × 32 cell, 每 cell 分到 1 act byte + 1 wt byte
               ↓ 每 cell 内部：
             MAC cell (L6) u8 × i8 乘法 (可配置精度)
               ↓ 一字节 weight → N 个 sub-byte → N 次 u8×sub-byte 累加
             Accumulator (L7) int32 acc[i][j] += product
               ↓ 全 K 累加完
             Convert (L8) × bias scale / 2, sat, → u16
               ↓ 写回 VTCM
             用户拿到 1024 u16 的输出 tile
```

从 L1 到 L8 共 8 层——这就是 v75 HMX 做一次 `u8 × i8` 32×32×32 MAC 的
"全景图"。每一层都是软件决策点：L1/L2 决定指令发什么，L3 决定数据
怎么摆，L4 决定 credit 密度，L5/L6/L7/L8 是硬件，但 L4 的 packet 序列
决定硬件怎么被利用（double-buffered readback、pre-pack、dual-scale 等
技巧都在这一层发挥）。

## 参考

- `Agent/hmx_int4_combos_analysis.md` — int4 / int2 / int1 / fp16 的并列分析
- `example/hmx_matmul_int16/int16_matmul_hmx.c` — bit-exact 参考实现
- `example/hexagon_hmx_matmul_native_int.md` — tile 布局实测日志
- `Agent/int4_matmul_optimization_log.md` — QNN 内置 kernel 的结构 reverse-engineering
