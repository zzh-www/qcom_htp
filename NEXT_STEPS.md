# V8C8 matmul — Step 1+2 完成；新发现：HMX kernel 5× gap 根因 = DSP-driven (2026-04-28)

> **重大更新 (2026-04-28)**: chrometrace 单事件级测量 (`optrace_compare_256_indep/`)
> 揭示 V8C8 BbbKMajor 与 native ConvLayer_s1.opt 在**纯 HMX kernel cyc**上差
> **5.3×** (V8C8 ~8000 vs native ~1500 per 256³ matmul)。之前 profile_text
> 的 1.23× 是 wall-time bucket 包了 HVX 并行时间的误导。
>
> **cyc/packet 一样**, HMX 物理速度同。**差距全在 DSP-issued packet 数**
> (V8C8 ~1950 vs native ~350 = 5.6×)。Native HMX kernel 是 descriptor-driven
> autonomous, V8C8 是 DSP-driven。完整记录: `project_v8c8_dsp_vs_descriptor_kernel_2026-04-28.md`。
>
> **新优先路径**: Path A = dlsym `hmx_convbbb1x1_stride1` + 自实现
> `set_hmx_params_conv1x1` descriptor builder。详见 §"Future direction" 末尾。

> **Status**: BbbKMajor 内核已经在 native `q::ConvLayer_s1.opt` 的
> **MatMul_0 group cyc 1.04× （基本持平）**. Steady-state (iter 3) 256³:
> **26 µs vs native 18 µs = 1.28× wall, ~1.45× cyc**. Bit-exact preserved
> at 256/512/1024³ (all 1.38 M cells).
>
> **Plan philosophy**: 用户明确要求 "复刻"，不接受 workaround 类 fix
> （e.g. 回滚输出 factorization）。所有路径必须在保持 native shape 形态下闭合。
>
> Step 1 ("+3K bank conflict 真凶") 完成。剩 Step 2/3/4。

## 最新状态：什么已对齐 ✓

| | V8C8 BbbKMajor | Native ConvLayer_s1.opt | match |
|--|----------------|-------------------------|-------|
| user input/output shape | `[1, 1, S, S]` u8 ↔ `[1, 1, S, S]` u8 | same | ✓ |
| in[0] act dim       | `[1, 8, 32, 256]` u8q | same | ✓ |
| in[1] wt dim        | `[1, 1, S, S]` u8q | `[1, 1, S, S]` i8q | ✓ shape |
| wt VTCM byte order  | `[1, N_t, K_t, 1024]` (Step 2) | `[1, N_t, K_t, 1024]` | ✓ byte-1:1 |
| in[2] bias dim      | `[1, 8, 1, 64]` i32 | same | ✓ |
| out[0] dim          | `[1, 8, 32, 256]` u8q (Crouton_8) | same | ✓ |
| HMX MAC body        | 2 cyc / MAC body (4 packets) | 2 cyc / MAC body (4 packets, native too) | ✓ at silicon ceiling |
| HMX iter order      | mt-outer / nt-inner (no bank conflict) | (presumed same — empirical 1.04× cyc match) | ✓ |
| HMX descriptor      | rebuild every call | baked at graph_finalize | ✗ Step 3 |
| Pipeline w/ HVX     | serial (Input → bbb → Output) | fused MatMul_0 (HVX/HMX overlap) | ✗ Step 4 |
| bit-exact matmul    | 256³/512³/1024³ all 100% | (reference) | ✓ |

## 当前 cyc 拆解 (256³ iter 3, MT_OUTER default)

```
                            V8C8 cyc   Native cyc   Δ        fix path
─────────────────────────────────────────────────────────────────────────
Input + ForceFormat_Crouton  ~10K       ~4K        +6K     Step 4 (build_tile)
wt DMA (OpId_17)              ~0.2K     (folded)   +0.2K   Step 4
bias DMA (OpId_18)            ~2.6K     (folded)   +2.6K   Step 4
HMX matmul kernel (bbb)       ~9.8K    ~9.5K group  +0.3K  ✓ at parity
Output Reshape + DDR (OpId_3) ~5.7K     ~5.2K       +0.5K  ✓ aligned
─────────────────────────────────────────────────────────────────────────
Total                        ~28K      ~17.8K      +10K   1.45× cyc / 1.28× wall
```

**bbb 内核已经追平 native MatMul_0 group**。剩下 +10K 全在外围
（Input ForceFormat_Crouton 6K + 我们 explicit DMA 2.8K + Output 0.5K）。
Step 4 (build_tile 让 QNN 把 ForceFormat / DMA / matmul 调度成
pipeline) 是最大单票。

## Step 1 已完成 — bank conflict 根因 + 修法

### 根因（已实证）

256³ 输出 Crouton_8 logical `[1,8,32,256]` 分配 32 blocks × 2 KiB。HMX
`:after:cm:sat.ub` 1 KiB write 占半个 block。当循环顺序为 nt-outer /
mt-inner 时, mt=0/1, 2/3, 4/5, 6/7 各对相邻 mt 写**同一 block** 的 offset 0
和 offset 1024，间隔仅 ~30 cyc（一次 K-sweep）。VTCM cache line 还在
飞→ +3K cyc bank conflict。

### 修法（commit fba391b）

切换到 mt-outer / nt-inner。同一 block 的两次 1 KiB 写现在间隔
N_t 个 nt-sweep（256³ 时 ~640 cyc）, cache line 完全 settle 后再写。
代价：bias mxmem2 重载从 N_t 次变成 M_t × N_t 次 (256³ 多 280 cyc;
1024³ 多 ~5K cyc, 但取消的 bank conflict 在 1024³ 是 ~13K cyc, 净赚)。

### 实测 (iter 3 steady state)

| | NT_OUTER (旧) | MT_OUTER (新) | Δ |
|---|---|---|---|
| 256³ bbb_M0_N0 | 12,852 cyc | **9,831** | -23% |
| 256³ Total accel | 30,524 | ~28K | -8% |
| 256³ Wall | 27 µs | 26 µs | -4% |
| 256³ vs Native MatMul_0 group | 1.35× | **1.04×** ✓ | parity |
| 1024³ Total | (untested) | 642K cyc / 324 µs | bit-exact ✓ |
| 512³ Total | (untested) | 110K cyc / 63 µs | bit-exact ✓ |

## 剩余路径 (ROI-ordered) — 全部 "复刻" 路径，无 workaround

### Step 2 — wt N-outer layout (DONE, commit 55915cb)

wt VTCM 字节序现在与 native q::ConvLayer.opt.weights_to_vtcm 完全一致
(`[1, N_t, K_t, 1024]`，tile (nt, kt) at `(nt*K_t + kt)*1024`)。

**关键负面发现**: 原计划把 `r8 = add(r8, 0x400)` 与 `weight.b =
mxmem(r8, r25)` 塞同一 packet (3-packet body / 1.5 cyc/MAC)。
**Hexagon V75 HMX 硬件不允许这个组合** — assembler 接受，runtime
SIGSEGV。

Native 实测 disasm (0x2ea830-0x2ea848) 也是 4-packet body：
```
Pkt 1: cmp + r26-=2 + memw r6 + memw r23
Pkt 2: r8 = add + 0xfd094718 + activation r6
Pkt 3: weight r8
Pkt 4: 0xe2198028 (likely r8 += 0x400) + activation r23
Pkt 5: weight r8 :endloop0
```
4-packet/2-MAC = 2 cyc/MAC body — **与我们一致**。handoff 中的
"3-packet/1.5 cyc/MAC" 是 disasm 误判。

**实测**: bbb 内核 cyc 没变（预期，inner asm 未改）；wt VTCM byte
顺序复刻达成（这是 Step 2 真正的 deliverable）。256³ wall 26 µs →
23 µs (≈1.28× → 1.22× vs native, 主要是 Output 段的 run-to-run noise)。

### Step 3 — `do_precomputation_function` (NOT VIABLE)

读 `qhpi.h:817-857` spec 实证：
- `QHPI_Plugin_Function function` 与 `QHPI_Plugin_Precomputed_Function
  function_with_precomputed_data` **互斥**（line 854: "When set, @c
  function must be NULL"）
- precompute 路径 runtime 函数签名 `(handle, precomputed_data)`
  **不接收 inputs/outputs** — 没法访问 act block_table

我们的 V8C8 kernel 必须 per-call 读 Crouton_8 act block_table（act
每次推理变），**precompute 路径架构上走不通**。Native 能用是因为
ConvLayer_s1.opt 是 QNN 内部 HTP primitive 不走 QHPI v1，特权高。

**Status**: NOT VIABLE — dropped。

### Step 4 — `QHPI_BuildTileOfOp` (DEFERRED, 不属复刻)

User memory 实测 `project_native_256_isolation_2026-04-26.md`:
> 256³ u8×i8 MatMul = exactly 1 ConvLayer_s1.opt (no graph scheduling).

**Native 256³ 也是单个 ConvLayer_s1.opt 不切 tile。我们当前 V8C8 256³
也是单个 BbbKMajor — 结构已与 native 一致。** Step 4 实施 build_tile
会让 QNN 把 BbbKMajor 切成多个 sub-op，反而**偏离** native 在该 shape
的真实形态。是"超越"native 而非"复刻"。

实施代价：
- Kernel 必须从假设方阵改成支持非方阵 (M≠K≠N)
- `qhpi_tensor_shape()` 在 ≥64³ 返回 rank=0，要新 shape 推导路径
- SDK 无 build_tile 参考实现（grep .so/.h/examples 全空）
- 写完整 slice 逻辑 + qhpi_op_create + 处理 wt 字节布局与 logical-slice
  不匹配

预期收益：handoff 估 3-5K cyc at 256³，但 256³ VTCM 充足无切分必要，
native 自己在该 shape 也不切分 — 收益不确定。

**Status**: DEFERRED — 复刻路径完成，需要时可作为下一步独立优化。

## ROI 表

| Step | what | 实测 cyc | effort | 状态 |
|---|---|---|---|---|
| 1 | mt-outer iter 杀 +3K bank conflict | -3K (256³) | medium | ✓ done (fba391b) |
| 2 | wt N-outer (byte alignment) | 0 | medium | ✓ done (55915cb) — handoff 中 1.5 cyc/MAC 是误判 |
| 3 | precompute hook | NOT VIABLE | — | ✗ dropped (架构不兼容) |
| 4 | build_tile callback | UNCERTAIN | high | ⏸ deferred (反向 native 256³ 单 op 形态) |
| **5** | **dlsym hmx_convbbb1x1_stride1 + 自实现 descriptor builder** | **预期 -6.5K cyc / matmul (5×)** | high | **NEW MAIN PATH (2026-04-28)** |

**Step 1+2 复刻路径完成。**但 chrometrace 单事件级测量（2026-04-28）揭示 V8C8
BbbKMajor 与 native ConvLayer_s1.opt 在**纯 HMX kernel cyc** 上差 **5.3×**
(不是 profile_text 的 1.23×)。根因: V8C8 是 DSP-driven (软件 loop0 显式发
~1950 packet 给 HMX), native 是 descriptor-driven autonomous (graph_finalize
烤 0x40 byte descriptor, runtime ~350 packet)。完整记录:
`Agent/.../project_v8c8_dsp_vs_descriptor_kernel_2026-04-28.md`。

## Step 5 — 切到 descriptor-driven autonomous HMX (NEW MAIN PATH)

> **完整执行计划 + Profile 验证方法**: `docs/v8c8_step5_descriptor_driven_plan.md`
> **架构洞见**: `docs/hmx_dsp_vs_descriptor_driven.md`

### 目标

V8C8 BbbKMajor steady-state HMX kernel cyc: 8,000 → 1,500 (5.3× 提速)。
256³ wall: 65 µs → ~25 µs (达到 native parity)。

### 路径

**Path A** (RECOMMENDED): dlsym 调 native `hmx_convbbb1x1_stride1` + 在
SkelOp 里自构 descriptor。

**已有基础设施**:
- `Agent/qnn_re/set_hmx_params_conv1x1.S` — 完整 disasm
- `Agent/qnn_re/hmx_convbbb1x1_stride1_full.S` — kernel disasm
- `project_dlsym_call_proven_2026-04-26.md` — dlsym 已验证可调通该函数 (1024-MAC test)
- `Agent/v8c8_alignment_phase2_BREAKTHROUGH_2026-04-27.md` — Crouton_8 框架已通

**剩余工作**:
1. 完整反逆 `set_hmx_params_conv1x1(out, arg1..5)` 的 5 个参数语义
   (5 个 packed-flag uint32, 编码 K/N/M/output_stride/mode_flags)
2. 在 SkelOp 里 per-call 算出 descriptor (shape-derived, 可缓存到栈)
3. dlsym `hmx_convbbb1x1_stride1`, 用我们的 descriptor + Crouton_8 act/wt/bias
   VTCM 地址调用
4. 调通 256³ bit-exact + 512/1024
5. 测 chrometrace 单事件 dur 应降到 ~1,500 cyc 与 native 一致

**风险**:
- VTCM stride / wt byte layout 必须严格匹配 native kernel 的期望 (1024-MAC
  spike 跑通过, 但 256³ 全 sweep 会暴露其他对齐约束)
- descriptor 字段语义反逆有风险, 错一个字段 HMX 直接 fault

**Path B** (备选): 自己模仿 native 写 mxdescriptor 配置, 不调 native 函数。
风险更高 (整个 HMX 启动序列要复现), 不推荐为首选。

**Path C** (兜底): 不动架构, 减软件开销 (移除 prebake、密集化 hardware loop)。
预期省 ~12% (~1K cyc), 不接近 5× gap。可作为 Path A 受阻时的 fallback。

### 完成 Step 5 后剩余 gap

Step 5 把纯 HMX kernel cyc 拉到 native parity (1,500 cyc/matmul) 后, V8C8 vs
native wall 应近 1.0× (剩余只是少量 DMA / Output 的 noise)。Step 4 (build_tile)
的 VTCM 切片 + HVX/HMX overlap 在 256³ 上不再必要 (native 自己也不切)。

## Per-shape 现状 (MT_OUTER default, 2026-04-27 night)

| S | V8C8 µs | bbb cyc | Total cyc | bit-exact | vs Native (256³ baseline ratio 1.28×) |
|---|---|---|---|---|---|
| 256 | 26 | 9,831 | ~28K | ✓ 65536/65536 | 1.28× wall |
| 512 | 63 | 59,431 | 110K | ✓ 262144/262144 | (待 native 实测) |
| 1024 | 324 | 462,896 | 642K | ✓ 1048576/1048576 | (待 native 实测) |

## 已知留存问题 (与本期 scope 无关)

- **S=128**: ~30% bit-exact (output Crouton_8 block_size=512 vs ≥256³ 的
  2048, HMX 1024B write overflow)。需 4th VTCM scratch input。
- **S∈{32,64}**: scalar fallback, 输出侧同根因。

## Build / run / verify

```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh

cd standard_flow/phaseB_v8
M=256 K=256 N=256 OUT_DIR=phase1_validation/v8c8_test bash run_v8c8_phase2.sh

# Profile decode (need to pull device qnn-profiling-data_0.log + run profile-viewer)
ssh oneplus "cat qnn_run/phaseB_c8/out/qnn-profiling-data_0.log" \
  > phase1_validation/v8c8_test/profile_dev.log
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
  --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so \
  --input_log phase1_validation/v8c8_test/profile_dev.log \
  > phase1_validation/v8c8_test/profile.txt
awk '/Number of HVX threads used : 4  count/{n++} n==3' phase1_validation/v8c8_test/profile.txt
```

## Reference (Agent/ + memory)

- `~/.claude/.../project_v8c8_dsp_vs_descriptor_kernel_2026-04-28.md` — 5× gap 根因 + Step 5 路径
- `~/.claude/.../reference_op_profiling_methodology.md` — N-instance independent profiling 方法
- `optrace_compare_256_indep/` — V8C8 vs native chrometrace 单事件级数据 bundle
- `Agent/qnn_re/set_hmx_params_conv1x1.S` — Step 5 必读: descriptor builder disasm
- `Agent/qnn_re/hmx_convbbb1x1_stride1_full.S` — Step 5 必读: HMX kernel disasm
- `Agent/dlsym_spike_PASS_2026-04-25.md` — Step 5 基础: dlsym 已验证
- `Agent/v8c8_alignment_phase2_BREAKTHROUGH_2026-04-27.md` — Crouton_8 框架进展
- `Agent/SESSION_2026-04-27_handoff_v8c8_alignment_status.md` — 前期 alignment audit
- `Agent/qnn_re/weights_to_vtcm_RE_2026-04-27.md` — wt DMA 字节布局
- `Agent/qnn_re/bias_to_vtcm_decoded_2026-04-27.md` — bias DMA 字节布局

## Recent commits

```
55915cb V8C8 Step 2: wt VTCM layout = native N-outer (复刻 alignment, no asm change)
7f685ad NEXT_STEPS: Step 1 done — kernel at native MatMul_0 group parity (1.04×)
fba391b V9_KERNEL_HMX: swap to mt-outer/nt-inner — kills +3K bank conflict (Step 1)
49f7aa7 NEXT_STEPS + handoff doc: V8C8 256³ kernel I/O fully aligned, 1.50× wall residual
cef7e01 gen_v8c8_test.py: align BbbKMajor I/O shapes with ConvLayer_s1.opt
356f40f gen_v8c8_test.py: align user-facing input shape with native [1,1,M,K]
2928716 run_v8c8_phase2.sh: enable optrace in ctxgen → schematic.bin emitted
eb6fd20 V8C8 BbbKMajor: 256³ matmul aligned to 1.33× of native
```
