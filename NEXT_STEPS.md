# V8C8 matmul — Step 1+2 done, 1.28× wall / wt byte-1:1 native (2026-04-27 night, post-Step-2)

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

### Step 3 — `do_precomputation_function`

Native 在 `set_hmx_params_conv1x1` 把 0x40 字节 descriptor 烤进
ctx-binary（graph_finalize 时一次过）。我们每次 call 都 prebake
act_ptrs/wt_ptrs (2 × 32 × 32 × 4 B = 8 KB) + 重设 HMX 状态。
QHPI 提供 `QHPI_Precompute_Function`，可把 shape-only-dependent 数据
烤进 `precomputed_data`, runtime kernel 仅读取。

**预期收益**: 1-2K cyc/call。

**Open question**: precompute 上下文里能否拿到 STATIC tensor `raw_data`?
如果只能拿 wt_ptrs 那样 shape-derived 信息（不能读 wt 字节）, 收益减半。
读 `qhpi.h:639-658` spec; 不清楚就写 probe。

### Step 4 — `QHPI_BuildTileOfOp` (build_tile callback)

Native 把 bias_to_vtcm + ForceFormat_Crouton + weights_to_vtcm +
ConvLayer_s1.opt fuse 在一个 `MatMul_0` group 下，QNN scheduler 把 HVX
(ForceFormat) 与 HMX (matmul) 并发调度（不同物理单元）。我们目前是
custom op, opaque to QNN, ForceFormat 串行跑在 Input 段。

实现 `build_tile` 把 BbbKMajor 按 N 切成 N_t sub-op。每个 sub-op + 其
周围的 DMA/format 就 pipeline-able。要写 `qhpi_op_slice` + `qhpi_op_create`
在 prepare time 构造 sub-graph。

**预期收益**: 3-5K cyc。最大单票，最高难度。

**先决条件**: Step 2/3 完成且 bit-exact 验证仍通过。

## ROI 表

| Step | what | 预期 cyc 节省 (256³) | effort | 风险 | 状态 |
|---|---|---|---|---|---|
| 2 | wt N-outer (byte alignment, 无 cyc 收益) | 0 | medium | — | ✓ done (55915cb) |
| 3 | precompute hook | 1-2K | medium | qhpi.h spec 不清楚 | next |
| 4 | build_tile callback | 3-5K | high | 复杂度高, payoff 不确定 | last |

完成 3+4 预期: 1.28× wall → ~1.10×。

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

## Reference (Agent/)

- `SESSION_2026-04-27_handoff_v8c8_alignment_status.md` — 前期 alignment audit
- `v8c8_step3_crouton8_output_2026-04-27.md` — Crouton_8 输出
- `v8_c8_kernel_perf_hwloop_2026-04-27.md` — hw loop0 + pre-baked ptrs
- `qnn_re/weights_to_vtcm_RE_2026-04-27.md` — wt DMA 字节布局
- `qnn_re/bias_to_vtcm_decoded_2026-04-27.md` — bias DMA 字节布局
- `qnn_re/hmx_convbbb1x1_stride1_full.S` — native HMX kernel disasm
- `qnn_re/set_hmx_params_conv1x1.S` — native descriptor builder disasm

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
