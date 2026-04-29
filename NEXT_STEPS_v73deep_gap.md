# Next Steps — V73DEEP → Native Gap @ 256³

## 当前状态（2026-04-29 — P1.4/P1.5 后）

| Path | dur (cyc) | pkts | cpp | bit-exact | vs native |
|------|----:|-----:|----:|:---:|:---:|
| **V73DEEP + native descs + native LAYOUT + HVX 表 copy**（**当前最佳**） | **1799** | **471** | 3.82 | 100% | **1.61× cyc / 1.36× pkts** |
| V73DEEP main path（旧 baseline，2026-04-28） | 3042 | 747 | 4.07 | 100% | 2.71× cyc / 2.16× pkts |
| Native q::ConvLayer_s1.opt | 1120 | 346 | 3.24 | ref | 1.0× |

**Build flag set**:
```bash
DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST"
DEFS="$DEFS -DV73D_N_TILES_POW2=32 -DV73D_OUT_Y_STRIDE=32 -DV73D_AD_ACT_Y_STRIDE=32"
DEFS="$DEFS -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1 -DV73D_NATIVE_LAYOUT=1"
EXTRA_DEFS="$DEFS" bash build.sh && bash build_x86.sh
```

**P1.4/P1.5 通过 patch native skel.so 抓 ground truth（详见
`Agent/qnn_re/p1_4_kernel_patch_2026-04-29.md`, `p1_5_native_layout_2026-04-29.md`,
和 `reference_vendor_kernel_patch_dump.md`）已确定**：
- 5 个 descriptor 差异（n_tiles_pow2=32, strides=32, mask args, mask[+0x38]=extra ptr）
- act_tbl/out_tbl 必须是 32 entries × 0x800 (2KB) M-pair packed 布局
- HVX vmem 一次性 copy 32-ptr table 比 scalar memcpy 省 ~140 pkts
- Gap 由 2.16× 关到 1.36×（37% packet 减少, 100% bit-exact）

**测量方法已固化**：`scripts/perf_v8c8.py <run> --compare <native>`，详见
`docs/v8c8_perf_reading_method.md`。

**指标语义**: chrometrace pkts ≈ PMU `COMMITTED_PKT_ANY`（不是 T0）。

## 已确认（不要再重做）

### 已 ruled out（empirical sweeps）

| 方向 | 试过的范围 | 结论 |
|------|------------|------|
| mask args (set_hmx_params_conv1x1) | arg1 ∈ {0x700, 0x70b, 0x73b, 0x780}; arg4 ∈ 0..0x40; arg5 ∈ {0x20, 0x21, 0x60, 0xa0, 0x120} | 全 PASS 但 pkts 永远 747 |
| 描述符字段缩小 | n_tiles_pow2 / k_total_bytes / n_act_pairs ↓ | pkts 等比下降，输出覆盖率也等比下降，无 magic 组合 |
| m_total_minus_step ↑ | 16, 32 | 反而 K 多算一遍，PASS 但 pkts 上涨 |
| VTCM 显式 8MB | vtcm_mb=8 | 无变化 |
| 替换 kernel entry | unaligned / NxN-bbb / OLD-unaligned | SIGSEGV (ABI 不同) |
| 多 call 切分 | 4 calls k_total=64 / 8 calls per-M-tile | 反而更慢（call 开销 dominant） |
| OLD kernel 单 call + V73DEEP 描述符 | — | crash |
| V73 non-deep + Lane A v2 | — | 1.74× 慢于 V73DEEP |

### 已 verified

- 同一个 dlsym kernel symbol：`hmx_v73_convbbb1x1deep_stride1` (0x2ebe40)
- 我们走 path 1（n_act_pairs > 1, extra_param[0] == 1）— 最 dense 的路径
- 我们的 mask_desc 字段值与 native 已通过 V9_PARAMS_PROBE + V9_DESC_DUMP 对齐
  （`Agent/qnn_re/set_hmx_params_conv1x1_probe_2026-04-28.md`,
  `v73deep_desc_dump_findings_2026-04-29.md`）
- chrometrace `pkts` ≈ PMU `COMMITTED_PKT_ANY`（不是 T0）

### 已知 gap 不在哪里

- 不在 mask 描述符
- 不在我们写的 out_desc / act_desc loop 字段
- 不在 wrapping（已经 ~50 pkts，Lane A 砍到底了）
- 不在 cpp（4.0 vs 3.24，差异只占 16%；剩余 84% 在 pkt count）

---

## 2026-04-29 session 跑过的方向

### 1. Wrapper 0x3dc2a8 静态 RE — **公式拿到，但有 vtable 卡点**

完整反汇编 `0x3dc2a8 → 0x3dc4b0` 和 `0x3d7920` 描述符 builder。

**结果：**
- 循环次数 `r23 = memw(memw(arg1+0x8) + 0x4)`
- 每 iter 推进 `r24 = sd0_tile_count * 4`（act 指针数组）
- 每 iter 推进 `r25 = sd1_tile_count * 4`（wt 指针数组）
- Tile-count 公式 `(sd[0x18]>>3) * (sd[0x1c]>>3) * (sd[0x20]>>5)`
- 描述符 builder 把 `od` (+0x10..+0x44) 和 mask area (+0x60..+0xa0) 写到 `r0+0x18` 区域

**卡点：** Builder 在 0x3d7b1c 走 `callr r2`（vtable[+0x24] 虚函数派发到张量对象），返回值喂给 `memw(r16+0x10) onward`。这条没解出，无法盲填。

**Output：** `Agent/qnn_re/wrapper_3dc2a8.S`、`wrapper_3dc2a8_TRACE.md`、
`descriptor_builder_3d7920_full.S`。

### 2. V9_DESC_DUMP 运行时 dump — **确认我们的描述符正确**

构造 V9_DESC_DUMP 编译模式，把我们 V73DEEP 路径的 mask/od/ad/extra_param/
act_tbl_all/out_tbl_all bytes 写进 Crouton_8 输出。Untile 后用 python parser 解码。

**结果：** 256³ 各字段 100% 符合 kernel 期望（n_tiles_pow2=64→r20=8,
k_total_bytes=256→r13=8, n_act_pairs=8→loop0 trip=4 等）。

**结论：** Gap 不在描述符 calibration 上。

**Output：** `Agent/qnn_re/v73deep_desc_dump_findings_2026-04-29.md`、
`scripts/parse_v73deep_desc_dump.py`。

### 3. V9_PROBE_REGIONS 4-variant 线性拟合 — **Gap 定位 K-MAC body**

跑 4 个 kernel call variant（V0=baseline, V1 减 r20, V2 减 r28, V3 减 r13e），
线性拟合：

```
P(r13e, r20, r28) = A + r13e * (B + r20 * (C + r28 * D))
```

**V73DEEP @ 256³ (PMU ANY units)：**
- A = 396 (prologue/epilogue)
- B ≈ 0 (per-outer overhead)
- C = 8.9 (per-loop1 drain)
- **D = 8.4 (per-loop0 K-MAC body — 74% of 1759 ANY total)**

**V73 non-deep 跨 kernel 对比：** D=11.55, V0=2407 — 比 deep 还差。
**OLD kernel：** K-major wt 不兼容，crash。

**结论：** 所有公开的 bbb (u8 输出) HMX kernel 每 K-MAC pair ≥ 8 ANY pkts。
Native 必须用 ~4 ANY/K-MAC pair 才能达到 346。

**Output：** `Agent/qnn_re/v73deep_pmu_regions_2026-04-29.md`、
`scripts/parse_v73deep_probe_regions.py`、V9_PROBE_REGIONS / V9_PROBE_V73_NONDEEP /
V9_PROBE_OLD_KERNEL 编译宏。

### 4. V75 hbh 系列 disasm — **找到 `:2x2` cvt fan-out，但 bbb 用不上**

V75-specific 的 fp16 输出 kernel `hmx_v75_convhbh1x1deep_stride1` (0x2f5200) 用了
30 个 `:2x2` cvt 指令，是多 tile fan-out 的关键。

**结论：** 文档 `docs/hmx-programming-guide/09-instr-convert.md` 明确写道
"matmul 只用 `:2x1`"，"`:2x2` QNN 未用"。Bbb (u8 输出) kernel 完全没有 `:2x2`
变体。这条优化 dlsym 路径拿不到。

**Output：** `Agent/qnn_re/hmx_v75_convbbh1x1_stride1.S`、
`hmx_v75_convhbh1x1deep_stride1.S`、`hmx_v75_convhbh1x1_stride1.S`。

### 5. matmul_qu8xqi8_32 / matmul_qu8xqu8_32 — **HVX vrmpy 软件 matmul，无 HMX**

这两个 `matmul_*` 符号是 `vrmpy` 走 HVX 的纯软件矩阵乘，不是 HMX kernel。
"_32" 是固定 32 列宽，给小尺寸用的。无关。

### 6. P1 — ctx-binary 字节提取（2026-04-29）— **排除 "inline kernel" 假说**

完整执行：见 `Agent/qnn_re/p1_ctx_binary_extraction_2026-04-29.md`。

关键结果：
- **Native ctx-binary 里零 `mxmem` 字节** — 不嵌内联 kernel；和我们一样 dlsym 公开符号
- Native 用 template `ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi` —
  description 明确指出："Conv2d ... extra CTRL input that controls
  whether we do a depthwise/grouped conv or use the **4/8/16bit
  kernels**"
- Native ConvLayer_s1.opt 输入 ops = **5** vs 我们 BbbKMajor = **3**
  （多的 2 个里有一个是 CTRL tensor）
- Hot 实例 chrometrace apples-to-apples 对比：

  | 指标 | Native MatMul_1 | 我们 bbb_chain1 (V73DEEP prod) |
  |------|---:|---:|
  | dur | 1446 cyc | 3042 cyc |
  | cpp | 4.18 | 4.07 |
  | pkts | **346** | **747** |

  **cpp 完全一致；gap 是纯 packet count (2.16×)。**

新 hypothesis：**Native 通过 ConvLayer_s1.opt dispatcher + CTRL input
触发 kernel 内一个我们没找到的 "fast path"**。可能 `*_unaligned` 变体
或某个特定 mask args 组合。

---

## 当前结论：dlsym 路径已穷尽

所有 3 个公开的 bbb HMX kernel（OLD / V73 / V73DEEP）每 K-MAC pair ≥ 3 mxmem
packets = ≥ 8 ANY committed packets。**V73DEEP 是公开符号能达到的上限。**
剩余 2× gap 的根源在 native 用了我们看不到的 kernel 或 inline 代码。

候选：
- libHtpPrepare 在 prepare-time 生成的内联汇编 kernel 直接嵌到 ctx-binary 里
- 或 HMX state-config 描述符驱动模式（指令编码不同）

---

## 还没试的方向（按性价比排序）

### ✅ P1 — Ctx-binary 字节提取（2026-04-29 完成）

P1 排除了 inline-kernel 假说。详见上面 §6 + `Agent/qnn_re/p1_ctx_binary_extraction_2026-04-29.md`。

P1 出来后剩下的子任务（按便宜→贵排）：

### ✅ P1.1 — `*_unaligned` 测试（2026-04-29 完成 — 关键发现，但 production blocked）

详见 `Agent/qnn_re/p1_1_unaligned_2026-04-29.md`。

**关键发现：** V73 unaligned kernel 用 **`:single:cm`** modifier (NOT `:deep:cm`)。
按 HMX 文档 `:single` 才是 1×1 matmul 的正经 modifier。但：

- Probe 显示 V0 = 829 ANY pkts（vs V73DEEP V0 = 1724，~2× 减少）。但 probe
  其他 variant 数字反常（V1=1048 > V0=829），很可能 kernel 在 degenerate
  输入下进入无限循环或越界，PMU snapshot 是 partial count。
- Production build (`V9_KERNEL_V73_UNALIGNED`) 在 chain=1 / chain=8 都
  SIGSEGV → unaligned 期望的 input pointer alignment 或 descriptor 字段
  语义和 deep 不同，没法 drop-in 替换。

**否定的假设：** "V73 non-deep with arg5=0 用 `:single` body" — 错。
`hmx_v73_convbbb1x1_stride1` (0x2eadc0) non-deep body 也用 `:deep:cm`。
`:single:cm` 只在 `*_unaligned` 变体里。

**真实结论：** 所有 `aligned` 1×1 bbb kernel (V73 / V73DEEP / OLD) 都用
`:deep:cm` 或 `:cm`，packet 数都 > 700 量级。**Native 346 packets 仍然
不在 dlsym 路径能访问的范围内。**

### ✅ P1.2 — Native CTRL input RESOLVED（2026-04-29 完成）

详见 `Agent/qnn_re/p1_2_ctrl_input_2026-04-29.md`。

通过 `s256_chain8_compare/model_schematic.bin` graph_after_optimization 段
解出 native ConvLayer_s1.opt 的 5 个输入：

- act / wt / bias — 和我们一样
- **CTRL = `0x08000000`** = int32 **8** → 选择 8-bit kernel
- **meta = `[1, 0]`** → 与我们 `extra_param[0..1] = {1, 0}` 完全一致

**重大结论：** 所有 dispatcher inputs 已对齐 native（CTRL=8, meta=[1,0]
match extra_param, mask 字段 match per V9_DESC_DUMP, od/ad 字段 match）。
但 packet 数仍然 747 vs 346 = 2.16×。

**Gap 一定在 dispatcher 内部我们看不到的地方** —— 最可能是 native dispatch
前发了 HMX state-config packets（HMX descriptor register 写入）我们没复现。

下一步必须 P1.3 (libHtpPrepare RE) 或 P1.4 (on-device kernel patch) 才能
继续推进。

### ✅ P1.3 — libHtpPrepare RE + 4-path empirical sweep（2026-04-29 完成）

详见 `Agent/qnn_re/p1_3_libHtpPrepare_RE_2026-04-29.md`.

**libHtpPrepare 部分**: 3 个 registry entries 引用 `ConvLayer_s1.opt@CB*2.Xsb.Fi.t.fi`
（2 primary + 1 fallback），primary 共享函数 `0xf1a360` 是 `hnnx::cost_func_from_str`
polynomial scorer，**不是** kernel dispatcher。仅引用 `hmx_v73_convbbb1x1_stride1`
非 deep entry，与我们走 deep 是同一 dispatcher 入口。

**Deep variant 4-path empirical sweep**:

| Path | N_ACT | EP[0] | hot pkts | bit-exact | vs native |
|---|---:|---:|---:|---:|---:|
| Main | 8 | 1 | 747 | 100% | 2.16× |
| Alt-A | 8 | 2 | 937 | 46% | 2.71× |
| **Alt-B** | **1** | **1** | **367** | **60%** | **1.06×** ✓ |
| Alt-C | 1 | 2 | 581 | 41% | 1.68× |
| Native | ? | ? | 346 | ref | 1.0× |

**关键发现**: alt-B (`-DV73D_N_ACT_PAIRS=1`) 367 pkts ≈ native 346 但 60% bit-exact——
每 loop0 iter drain 写早期 cells 只累加 partial K，最后 cell 才正确。证明 HMX
能 ~350 pkts 跑 256³ matmul，但 main path descriptor 调整无法触发。

**结论**: 4 个静态可达 path 都不是 native 的真正 path。剩余假设：
HMX state-config descriptor-driven 外部预写让 kernel 内 loop 早结束。
**必须 P1.4 (patch native kernel dump runtime args) 才能继续推进**。

### ✅ P1.4 — On-device kernel patch BREAKTHROUGH（2026-04-29 完成）

详见 `Agent/qnn_re/p1_4_kernel_patch_2026-04-29.md`.

**Method**: 156 bytes patched at offset 0x2ebe40 in libQnnHtpV75Skel.so
(deep variant entry). Replaced kernel with code dumping r0..r5 + mask + od +
ad + extra to first output tile. Pushed to device qnn_run/, ran native, pulled
output, decoded.

**Native's true descriptors** (5 关键差异 vs 我们 V73DEEP):

| 字段 | Native | 我们之前 |
|---|---|---|
| `mask[+0x0c]` (act_rt_base) | **0x71f** | 0x77c |
| `mask[+0x38]` | **= extra_param ptr** | 0 |
| `od.out_y_stride_words` | **32** | 0 |
| `od.n_tiles_pow2` | **32** | 64 |
| `ad.act_table_y_stride_words` | **32** | 0 |

Plus: native uses VTCM at `0xfc010000+` for outputs (0x800 stride), VTCM for
wt/bias too. We use DDR.

**Empirical (apply native descs)**:

| Config | hot pkts | hot dur | bit-exact | vs native |
|---|---:|---:|---:|---:|
| V73DEEP main path | 747 | 3042 | 100% | 2.16× |
| **+ native descs** | **475** | **1730** | **50%** | **1.37×** |
| Native | 346 | 1120 | ref | 1.0× |

**Gap reduced 2.16× → 1.37×**. 50% bit-exact 因为 n_tiles_pow2=32 让 kernel
只产 half M outputs。Native 通过 act_tbl_all/out_tbl_all 的 stride=32 dwords 布局
+ n_tiles_pow2=32 同时实现 full coverage AND 减半 packet count.

**剩余 1.37× gap → P1.5**: RE act_tbl_all 完整 layout (stride=32 dwords pattern)
+ implement VTCM placement for wt/bias.

### ✅ P1.5 — Native table layout RE COMPLETE（2026-04-29）

详见 `Agent/qnn_re/p1_5_native_layout_2026-04-29.md`.

P1.4 v3 patch dumped 32 act_tbl entries + 16 out_tbl entries within readable
tile region. Both: 32 entries × 0x800 (2KB) stride VTCM layout. Each ptr =
M-pair packed (2 M-tiles per 2KB region).

**Implementation**: V73D_NATIVE_LAYOUT knob builds 32-entry tables via HVX
vmem batch (1 HVX vector = 128 bytes = 32 ptrs × 4 bytes per table).

**Final result @ 256³ chain8 hot**:

| Config | hot pkts | hot dur | bit-exact | vs native |
|---|---:|---:|---:|---:|
| Baseline V73DEEP | 747 | 3042 | 100% | 2.16× |
| **+ P1.4 + P1.5 (native descs + layout + HVX)** | **471** | **1799** | **100%** | **1.36×** |
| Native | 346 | 1120 | ref | 1.0× |

**37% packet reduction with 100% bit-exact preserved**. Gap closure
2.16× → 1.36×.

**Build flags** (final best):
```bash
DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST"
DEFS="$DEFS -DV73D_N_TILES_POW2=32 -DV73D_OUT_Y_STRIDE=32 -DV73D_AD_ACT_Y_STRIDE=32"
DEFS="$DEFS -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1 -DV73D_NATIVE_LAYOUT=1"
```

**Remaining 1.36× gap**: ~125 packets. Sources:
1. VTCM placement (~80 pkts): native has wt + bias in VTCM (0xfc02_xxxx),
   we use DDR. Need TCM_Only signature in op-pkg.
2. Per-call kernel prologue (~45 pkts): native may pre-config HMX state
   once per session.

Closing further requires P6 (TCM_Only signature) — multi-day op-pkg work.

### 🟢 P2 — 提取出来的字节直接复制 + dlsym 调用（OBSOLETE — P1 排除了）

P1 已确认 native 不嵌 inline kernel。此选项无意义，删除。

### 🟡 P3 — NEXT_STEPS 原 Phase 2 (kernel-prologue patching)

Patch v73deep kernel 入口的前 8 packet 跳过 prologue（save state once → jump to
body），后续 N-1 次复用。能省 (N-1) × ~30 packet。

但根据 PMU probe 数据：A (prologue) = 396 ANY 在 1759 ANY 总数里只占 22%。
就算 prologue 完全省掉，也只能从 747 chrometrace pkts 降到 ~600。**够不到 native 346**。

价值有限，除非配合 P1/P2。

### 🟡 P4 — libHtpPrepare.so 静态 RE（多天）

x86 二进制，工具好。找到 prepare-time kernel code generator 即可拿到 ground truth。

**做法：**
1. 在 `tools/qnn-sdk/lib/x86_64-linux-clang/libHtpPrepare.so` 里 grep `ConvLayer*opt*`
   相关符号
2. 找 prepare-time 把 mxmem 字节生成进 ctx-binary 的代码路径
3. 那里就是 native 256³ kernel 的真正构造源

成本：多天。和 P1/P2 互补，但 P1/P2 更便宜先试。

### 🔴 P5 — 自己写 HMX state-config inline kernel（多周）

如果要彻底 close gap 自己写。HMX state 寄存器（控制内部 fan-out 等）公开文档
里没有；得从 HMX-programming-guide / hmx_dsp_vs_descriptor_driven 文档 + 反汇编
推断。极高成本，不确定 fan-out 优化能否给 u8 用。

### 🔴 P6 — 接受 V73DEEP 是 ceiling

标记 2× gap 为已知限制，归档当前 V73DEEP 实现，移到下一个性能瓶颈（如多 instance
> 2048³, prepare 时间, etc.）。

---

## 推荐执行顺序

**P1 已完成（2026-04-29）—— 排除 inline kernel 假说，找到 dispatcher 线索**。

下一步建议：

1. **P1.1 + P1.2 一起做**（半天工作）：
   - 测 `hmx_v73_convbbb1x1_stride1_unaligned` 变体（V9_PROBE_V73_UNALIGNED 宏）
   - dump native CTRL input tensor（ID `0x0000100200000008`）字节内容
   - 如果其中之一显示 native 走的是不同 kernel 路径，gap 就有解；否则
     转去 P1.3/P1.4

2. 如果 P1.1/P1.2 没结果，**P1.3 (libHtpPrepare static RE)** 是唯一能
   彻底解谜的路径。多天工作，但 x86 工具好。

3. 实在不想投入：**P6 接受 V73DEEP 是 stock-kernel ceiling**，关掉这个
   gap-closing 路径，把精力转去：
   - 多 instance ≥2048³ 性能优化
   - prepare 时间优化
   - 其他模型层瓶颈

P1 之前的方向（mask 字段 sweep / wrapper 静态 RE / 多 call 切分）信息密度已经
穷尽，不要再做。

---

## 测量速查（不要再凭印象）

```bash
# baseline V73DEEP build + run
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST" \
    bash build.sh && \
EXTRA_DEFS="..." bash build_x86.sh
WT_LAYOUT=kmaj CHAIN=8 OUT_DIR="$(pwd)/standard_flow/phaseB_v8/phase1_validation/<name>" \
    bash standard_flow/phaseB_v8/run_v8c8_chain.sh

# compare
source ../../scripts/env.sh
python3 ../../scripts/perf_v8c8.py \
    standard_flow/phaseB_v8/phase1_validation/<name> \
    --compare standard_flow/phaseA_native/s256_chain8_compare

# descriptor dump (确认我们的字段)
EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST -DV9_DESC_DUMP" \
    bash build.sh && bash build_x86.sh && bash standard_flow/phaseB_v8/run_v9_desc_dump.sh

# per-region PMU probe (gap localization)
EXTRA_DEFS="... -DV9_PROBE_REGIONS" bash build.sh ...
WT_LAYOUT=kmaj CHAIN=8 OUT_DIR="..." bash standard_flow/phaseB_v8/run_v8c8_chain.sh
python3 ../../scripts/parse_v73deep_probe_regions.py <out_dir>/device_out/out.raw

# baseline kernel-only PMU (chrometrace ≈ PMU ANY 验证)
EXTRA_DEFS="... -DV9_PMU_PROBE" bash build.sh ...
# decode 8×u32 LE in out[0..31]: pkt_kernel_ANY/T0/inst/disp/cyc/op_total_*
```

输出格式：
```
GAP: ours/native  cyc=X.XX×  pkts=X.XX×  cpp_ratio=X.XX×
```

---

# 后续计划（2026-04-29 P1.5 后）

## 当前剩余 gap 解构

cpp ratio 1.18× → kernel-internal 已基本对齐 native。
Packet count gap 1.36× = ~125 packets surplus。来源猜测：
- **VTCM placement** ~80 pkts（native 把 wt + bias 放在 VTCM 0xfc02_xxxx，
  我们用 DDR via `qhpi_tensor_raw_data`）
- **Per-call kernel prologue** ~45 pkts（native 可能 chain 内 share HMX
  state setup，我们每 call 都重做）

## P2 优先级（按 ROI 排）

### 🟢 P2.1 — 跨 shape 验证（半天）

**目的**：确认 P1.4/P1.5 的 build flag 在 512/1024/2048/4096³ 同样有效，
不仅仅 256³。可能在大 shape 下：
- mt_per_block 不同 → V73D_NATIVE_LAYOUT 的 32-entry HVX path 不触发
- 描述符 strides 需要重新对齐
- 性能比例与 256 不同

**做法**：
```bash
for SIZE in 512 1024 2048 4096; do
  M=$SIZE K=$SIZE N=$SIZE WT_LAYOUT=kmaj CHAIN=8 \
    OUT_DIR=".../v73deep_native_hvx_s${SIZE}" \
    bash standard_flow/phaseB_v8/run_v8c8_chain.sh
  python3 scripts/perf_v8c8.py .../v73deep_native_hvx_s${SIZE} \
    --compare standard_flow/phaseA_native/s${SIZE}_chain8_compare
done
```

如果某 shape gap 依然 >1.5×，patch native kernel 跑那 shape 的 dump（v3
patch 已 ready，只需换 OUT_DIR）。

### 🟡 P2.2 — VTCM placement for wt + bias（多天）

**预期 close ~80 pkts gap**，剩 ~45 pkts。

**做法**：在 `BbbKMajorOp.cpp` 的 op signature 里把 wt + bias declare 成
`Storage_TCM_Only`。QNN 编译器会自动在 op 前插入 `weights_to_vtcm` /
`bias_to_vtcm` ops（per
`reference_qnn_custom_op_scheduling.md` + `qnn_primitive_alignment_phase01_2026-04-26.md`）。

具体步骤：
1. 修改 `interface.cpp` 的 BbbKMajor signature: `inputs[1].storage = Storage_TCM_Only`
   for wt, `inputs[2].storage = Storage_TCM_Only` for bias
2. ctxgen 重新跑，确认 lowered graph 自动插入了 to-VTCM ops
3. 在 V9 op execute 里 wt_pack / bias_bytes 的 ptr 应该自动是 VTCM 地址（因
   QNN 的 indirect storage）
4. 测 chain8 hot pkts，预期 ~390 pkts 接近 native 346

**风险**：signature 改完 ctxgen 可能在 Graph Optimizations 阶段崩（per
P1.5 之前 V9_KERNEL_OLD_V73DESC 路径的经验）。需要 PASS 一次后才继续。

### 🟡 P2.3 — Per-call prologue elimination（高风险，多天）

**预期 close ~45 pkts gap**，到达 native 量级。

**机制**：native 在 chain 内可能多个 ConvLayer_s1.opt 共享 HMX state
寄存器配置（`mxmem` 写入 HMX descriptor 寄存器一次，后续 call 跳过）。

**做法**：
1. P1.4 v4 patch dump kernel prologue 阶段写入的 HMX state（前 30 packets
   写入了什么寄存器）
2. 把那些寄存器写入提到 V9 op 的 graph init/preCondition 阶段（per-graph
   一次而不是 per-call）
3. Patch local kernel 入口跳过对应 prologue packets

**复杂度**：高。要看是否值得 vs 接受 1.36× gap。

### 🔴 P2.4 — 直接接受 1.36× as ceiling（recommend）

V73DEEP 已经从 2.16× 关到 1.36×。dlsym 路径下能做的事我们都做了。
剩余 gap：
- 80 pkts in VTCM placement —— 需要 op signature 修改 + ctxgen 调试
- 45 pkts in kernel internal —— 需要更多 patch 实验

**P6 (NEXT_STEPS 原)更新版**: V73DEEP 已经是 dlsym 实战可达的 best。
推荐转去：
- **多 instance ≥2048³ 性能优化** — 大 shape 的 Spill/Fill 优化
- **prepare 时间优化** — ctxgen 加速
- **其他模型层瓶颈** — 比如 attention layer norm

## 推荐执行顺序

1. **P2.1 先做**（半天）— 验证 P1.4/P1.5 flag 跨 shape 有效。如果只 256
   有效，那 1.36× 就只是 256 一个点的 win，需要 per-shape 调优，价值大降。
2. **P2.1 全过 → P2.2** — VTCM placement 是 closing 80 pkts 的最大单一
   投入。signature 修改 + ctxgen 调试，2-3 天。
3. **P2.2 完成 → 评估**：到 ~1.13× 时再决定 P2.3 是否值得。如果 ~1.13×
   够好就接受。
4. **如 P2.1 在大 shape 失败，换 P2.4**: 接受 256 的 1.36× win，文档化，
   不再追 V73DEEP gap，转去其他瓶颈。

## 测试 / RE infra 已就绪

- `Agent/qnn_re/p1_4_kernel_patch_2026-04-29.md` — kernel patch + descs
- `Agent/qnn_re/p1_5_native_layout_2026-04-29.md` — table layout RE
- `reference_vendor_kernel_patch_dump.md` — patching method 总结
- `/tmp/p14_patch/` — patch tools (dump_stub_v1/v2/v3.c, .bin)
- 复现 patch dump：见 `reference_vendor_kernel_patch_dump.md` 的 recipe
