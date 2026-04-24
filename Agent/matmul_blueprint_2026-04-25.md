# MatMul blueprint — 照抄 QNN built-in 的正确写法 (2026-04-25)

> Based on:
> - `libQnnHtpV75Skel.so::hmx_convbbb1x1_stride1 @ 0x2ea740` 反汇编
> - `convert_to_crouton_b @ 0x237700` 反汇编
> - 512³ w8a8 实测 chrometrace (sweep_data_2026-04-19/s512/w8a8/)
> - 前期 RE 笔记: `qnn_hmx_pipelining.md`, `qnn_vs_v8_root_cause_2026-04-24.md`,
>   `forceformat_crouton_re.md`, `ours_vs_qnn_fundamental_diffs.md`

**V8 的 core insight 错了**: 我们以为差距来自 "HMX-HVX overlap"。实测 QNN 的
chrometrace 显示 **HMX 和 HVX 几乎不重叠** — HMX 在整个 timeline 66K cyc
里只活跃 12K cyc (18%)。QNN 的速度来自**图级切分 + HVX 并行 pack**，不是
overlap。

## 1. QNN 实测时间线 @ 512³ w8a8

```
  cyc     0    5K   10K  15K  20K  25K  30K  35K  40K  45K  50K  55K  60K  65K
HMX-256  ██ █ █                                                      ████ █ ██ ██    OutputSlice
HVX-513      ████████████████▓ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                          ▓
HVX-515     ████████████████▓ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                ████████████             ▓
HVX-512                                                                         █████████

          ▲bias+wt                          ▲                         ▲ 4× ConvLayer  ▲ Concat
           to VTCM                            pack done                MAC sequential   + OutputSlice
           (9.6K cyc)

 █ HMX event  ▓ HVX ForceFormat_Crouton  ▬ HVX InputSlicePad
```

**每个事件的确切数据：**

| tid  |  start |   end  |   dur  | op                             | dims                |
|------|-------:|-------:|-------:|--------------------------------|---------------------|
| 256  |     10 |   9618 |  9,608 | bias_to_vtcm ×2, weights_to_vtcm ×3 (prefetch)  | [1,1,512,256] |
| 513  |    482 |  24721 | 24,239 | **InputSlicePad** M_half0      | [1,8,32,512] (M=256, K=512) |
| 515  |    462 |  23799 | 23,337 | **InputSlicePad** M_half1      | [1,8,32,512] |
| 513  |  24781 |  44303 | 19,522 | **ForceFormat_Crouton** M_half0| [1,8,32,512] |
| 515  |  24813 |  44337 | 19,524 | **ForceFormat_Crouton** M_half1| [1,8,32,512] |
| 256  |  44386 |  49788 | **5,402** | ConvLayer_s1.opt (M0,N0)    | [1,8,32,256] |
| 256  |  49990 |  52261 | **2,271** | ConvLayer_s1.opt (M0,N1)    | [1,8,32,256] |
| 256  |  52277 |  54570 | **2,293** | ConvLayer_s1.opt (M1,N0)    | [1,8,32,256] |
| 256  |  54593 |  56864 | **2,271** | ConvLayer_s1.opt (M1,N1)    | [1,8,32,256] |
| 515  |  52324 |  63324 | 11,000 | OutputSlice                    |                     |
| 512  |  56928 |  66445 |  9,517 | OutputSlice                    |                     |
|      |        |        |        | **total wall-clock**           | **66,445 cyc**      |

**关键数字：**
- HMX MAC 总活跃：12,237 cyc = 4 × ConvLayer (5402+2271+2293+2271)
- HMX 利用率 (vs timeline)：12,237 / 66,445 = **18.4%**
- HVX pack 总活跃 (2 threads 并行)：43,761 cyc 每线程
- HVX overlap with HMX: ~0（pack 在 0..44K，HMX MAC 在 44K..57K，**串行**）
- HMX prefetch with HVX pack: ✓ (HMX 在 0..9.6K 做 bias/wt to VTCM，HVX 在 0.5K 就开始 pack)

## 2. MatMul 的正确拓扑

一个 `MatMul[M,K]×[K,N]→[M,N]` 展开成 **10+ 节点的子图**：

```
 input_flat[1,1,M,K]                         weights_flat[K,N] + bias[N]
       │                                              │
 ┌─────┴──────┐                            ┌──────────┼──────────┐
 │            │                            │          │          │
 ▼            ▼                            ▼          ▼          ▼
InputSlicePad(M0)   InputSlicePad(M1)    weights_to_vtcm(N0)  weights_to_vtcm(N1)
[HVX t513]          [HVX t515]           bias_to_vtcm(N0)    bias_to_vtcm(N1)
 │                   │                    [HMX t256, prefetch, parallel to HVX pack]
 ▼                   ▼
ForceFormat_Crouton  ForceFormat_Crouton
[HVX t513]           [HVX t515]
 │                   │
act_C_M0             act_C_M1
 │                   │
 └─┬─────────────────┘─────────────────┬───────────┐
   │                                   │           │
   ▼                                   ▼           ▼
 ConvLayer_s1.opt(M0,N0)  (M0,N1)   (M1,N0)   (M1,N1)    ← 4 MAC rounds
 [HMX t256, sequential through single HMX unit]
   │                    │              │         │
   └──── Concat(N) ─────┘              └─ Concat(N) ─┘
           │                                    │
           └───── OutputSlice ──────────────────┘
           [HVX t512/515, unpack Crouton → DDR [M,N] row-major]
```

**分片维度：**
- `M` 方向切 2 half（每 half 256 行）→ 两个 HVX 线程各处理一个 M_half 的 pack
- `N` 方向切 2 half → 每个 M_half 对应 2 个 ConvLayer_s1.opt 调用（N_half0, N_half1）
- 结果 4 个 `[M_half, N_half]` 象限 → 2 个 Concat（每个合并 1 个 M_half 的 N 维）→ OutputSlice 合并 M 维 + 转 DDR 布局

## 3. HMX 内核（`ConvLayer_s1.opt`）是什么样子

完全反汇编出来的热循环，**就是我们 V8 的 MatMulV8 已经在跑的形状**：

```asm
; outer loop1 (per output tile):
  r3 = r3 + 0x100
  bias = mxmem2(r3)                      ; bias load
  mxclracc                               ; clear acc

  ; inner loop0 (K/2 iters, 2-MAC unroll):
  { activation.ub = mxmem(r6, r24):cm    ; Rt_act = 2063 (2047|0x1C)
    weight.b      = mxmem(r8, r25) }     ; Rt_wt  = 0x3FF  ← KEY, 2.5× speedup
  { activation.ub = mxmem(r23, r24):cm
    weight.b      = mxmem(r8, r25) }

  ; post-K:
  r10 = memw(r0++m0)                     ; next output tile ptr from list
  mxmem(r10, r11):after:cm:sat.ub = acc  ; WRITE 1 KiB contiguous to tile
```

**要点：**
1. `Rt_wt = 0x3FF`（**不是** 2047）—— 单独这个就给 2.5× speedup。V8 已经这么做。
2. `:cm` 在 activation 上，`:after:cm:sat.ub` drain。V8 也已经是这样。
3. `:after:cm:sat.ub` 直接写到 **tile-layout output** (1 KiB per tile contiguous)，
   **不是** row-major `[M,N]` scatter。V8 已经改成了这个。
4. 2-MAC unroll 是**循环展开**而非 pipelining。V8 已经这么做。
5. **HMX 单线程** (`tid=256`) — QNN 没有"多线程 HMX"。一个 HMX unit，串行跑 4 次。

**V8 的 HMX 核心已经和 QNN 等价**，per-ConvLayer-call 差距只在框架 overhead。

## 4. HVX 并行是 pack，不是 requant

QNN 的 HVX 线程（`tid=513` 和 `tid=515`）在干什么：

- **InputSlicePad**（24K cyc/thread）：从 DDR flat `[1,1,M,K]` 里取出自己负责的
  M_half（256×512 bytes），zero-pad（因为 HMX 要 32-对齐），落到 VTCM。
- **ForceFormat_Crouton**（19.5K cyc/thread）：把上一步的 flat 转成 HMX 原生的
  Crouton 布局（stride-32 depth-32 lane）。核心是两遍
  `V6_vshuffvdd(Vu, Vv, Rt=-32)`（byte 版）或一遍 `Rt=-2`（halfword 版）。
  **没有 LUT、没有 vdelta**，纯 vshuff。详见 `forceformat_crouton_re.md`。

这两个 HVX op **不依赖 HMX**，可以和 HMX 的 `weights_to_vtcm/bias_to_vtcm` 并行。
在 chrometrace 里：HMX tid 256 在 0..9.6K 做 prefetch，HVX 两个 tid 在 0.5K
就已经开跑 pack，所以 HMX prefetch 被 HVX pack **完全隐藏**。

## 5. **V8 照抄 QNN 需要做什么** — 差距拆解 & ROI

把 V8 现状（qnn-net-run 跑出来的 chrometrace）和 QNN 对比：

| 项                         | V8 (custom OpPkg)  | QNN built-in      | V8/QNN |
|----------------------------|-------------------:|------------------:|-------:|
| HMX timeline cycles        |         317,009    |         66,513    | 4.8×   |
| HMX active cycles          |         206,652    |         12,237    | 16.9×  |
| pack_act total cycles      |         520,132 (单线程 HVX) | 43,761 × 2 threads = 87,522 wall-clock but 2-way parallel | ... |
| 并行 HVX 线程数 (pack)     |              1     |              2    | —      |
| M 切分数                   |              1     |              2    | —      |
| N 切分数                   |              1     |              2    | —      |
| ConvLayer_s1.opt 调用次数 |           1 (mmv8) |              4    | —      |
| pack→MAC overlap           |  无（严格串行）    |  无（严格串行，但 HVX 并行）| —|

### 要追平 QNN，按 ROI 排序的 4 件事

#### ① 拆图：M 切 2 half + N 切 2 half（**最大收益 ~3×，必做**）

**现在：** V8 是一个巨 op (`pack_act → pack_wt → mmv8 → tcm2ddr`)，单线程全跑。

**照抄 QNN：** 标准 flow 里的 ONNX 改写成**多节点图**：

```python
# gen_v8_onnx.py 改写示意
N_SPLIT = 2
M_SPLIT = 2  # 对 M 不切也行，先只切 N 验证

# M0 pack
pack_act_M0 = Node("PackActivationU8RowMajor", ["act_M0"], ["packed_act_M0"])
# M1 pack (如果切 M)
pack_act_M1 = Node("PackActivationU8RowMajor", ["act_M1"], ["packed_act_M1"])

# N0 wt pack (each on different thread — custom op marked MT)
pack_wt_N0 = Node("PackWeightToHmxTileV3", ["wt_N0"], ["packed_wt_N0"])
pack_wt_N1 = Node("PackWeightToHmxTileV3", ["wt_N1"], ["packed_wt_N1"])

# 4 MatMul MAC rounds
mmv8_00 = Node("MatMulV8", [packed_act_M0, packed_wt_N0, bias_N0, scratch], [out_M0N0])
mmv8_01 = Node("MatMulV8", [packed_act_M0, packed_wt_N1, bias_N1, scratch], [out_M0N1])
mmv8_10 = Node("MatMulV8", [packed_act_M1, packed_wt_N0, bias_N0, scratch], [out_M1N0])
mmv8_11 = Node("MatMulV8", [packed_act_M1, packed_wt_N1, bias_N1, scratch], [out_M1N1])

# Concat N + copy to DDR
concat_M0 = Node("Concat", [out_M0N0, out_M0N1], [out_M0])
concat_M1 = Node("Concat", [out_M1N0, out_M1N1], [out_M1])
concat_full = Node("Concat", [out_M0, out_M1], [out_tile_full])
tcm2ddr = Node("TcmDramCopy", [out_tile_full], [out])
```

**前提 #1：pack ops 要加 `multithreaded=true`**。我们的 `pack_act_rm_hvx.c`
和 `pack_wt_v3_hvx.c` 已经是 MT=true（看 `sig_kernels[].multithreaded`），
并用 `qhpi_num_slices/qhpi_slice_number` 自切 M_tiles。但现在**跑单线程**，
因为 M_tiles 只有一个 pack_act 实例能用到。切成 2 个 pack_act 实例后，
QNN scheduler 自然会把它们发到 2 个 HVX 线程。

**前提 #2：4 个 MatMulV8 是否能并发？** —— 看 QNN 实测，4 次 ConvLayer_s1.opt
是**串行**在 HMX tid 256 上的。所以 4 个 MatMulV8 也会串行 —— 但时间
budget 会从现在 mmv8=363K cyc 降到每次 ~90K cyc × 4 = 360K，基本持平。
真正的速度提升来自 pack 被并行化。

**预期增益**：pack_act 520K → ~260K wall-clock（2 线程并行切 M）+ ~2×
pack_wt 并行（if applicable）= HMX idle 时间大幅缩短。

#### ② 让 HMX 在 pack 的时候 **预载 weights 到 VTCM**（~1.2×）

QNN 的 `weights_to_vtcm` / `bias_to_vtcm` 是 HMX thread 的 prefetch op，
跑在 HVX pack 之前就 done。V8 现在**没有**这个 prefetch 节点 —— wt_raw 是
STATIC，但 DDR→VTCM 的搬迁发生在 `mmv8` 里（隐式）。

**照抄方法**：把 `pack_wt` 的 QHPI kernel signature 从 `QHPI_RESOURCE_HVX`
改成标 `QHPI_RESOURCE_HMX`（或新增一个 pure-DMA op 用 HMX resource），
让 QNN scheduler 把它发到 HMX tid 256 跑 —— 这样它就能和 HVX pack_act
并行。具体 QHPI API 是否支持待验证；可能需要新 op type。

#### ③ 切掉 tcm2ddr，让 MatMulV8 输出直接是图的最终 output（~1.1×）

当前：`mmv8 → out_tile (VTCM) → tcm2ddr → out (DDR)` — tcm2ddr 额外 31K cyc。

QNN：`ConvLayer_s1.opt → Concat (VTCM) → OutputSlice → DDR`，OutputSlice 的
9.5K cyc 里**同时做了 Crouton → row-major 的 untile + DDR 拷贝**。

**照抄方法**：把 `UntileToRowMajor` op 替代 `TcmDramCopy`（我们已经留了这个
kernel 在 `kernel/untile_to_rowmajor_hvx.c`），并让 QNN 在 DDR write 时
用 HVX 大宽带拷贝（已经是这样，但没验证是否 coalesce）。

#### ④ 让 pack_act 跑多线程（~1.8×，已经接近可用）

V8 pack_act 现在跑 520K cyc 单线程。QNN ForceFormat_Crouton 每线程 19.5K，
两线程并行即 wall-clock 19.5K。差距 27×。两层原因：

- **我们的 pack_act 每 tile 慢 ~15×**：QNN 的 `convert_to_crouton_b`
  一次吃 4 行 × 128 列 (= 4 vmemu + 4 vshuff = ~8 HVX insns)；我们的
  `pack_one_rm_tile` 是 32 行 × 32 列用 scalar u64 loads (8 iterations × 16
  u64 ops = 128 scalar insns)。每 tile 我们 ~128 cyc，QNN ~8 cyc → 16×。
- **QNN 还切 M 成 2 路，我们是 1 路**：再 2×。

**照抄方法**：用 `V6_vshuffvdd(Vu,Vv,-32)` 两遍改写 `pack_act_rm_hvx.c`，
按 `forceformat_crouton_re.md` §4 的拓扑（4 rows × 128 cols at once）。
这是 `hvx_4way_byte_transpose_re.md` 里已经有设计的 HVX pack。

## 6. 一句话结论

> **不要让 MatMul 是一个 op。让它是 ~10 个 op 组成的子图，**
> **pack 用 `V6_vshuffvdd` 切 M 两并行，MAC 切 N 两串行，**
> **HMX 内核就和我们现在 V8 一样 `:after:cm:sat.ub` + `Rt_wt=0x3FF`。**

HMX 内核我们已经做对。Pack kernel 要彻底换成 QNN 那套 vshuff 拓扑，
而且必须**在 ONNX 图里切分成多实例**让 QNN scheduler 拆到多个 HVX 线程。

## 7. 实际动手顺序（按优先级）

| # | 动作                                              | 代码位置                                                   | 预期收益 |
|---|---------------------------------------------------|------------------------------------------------------------|----------|
| 1 | ONNX 里 pack_act 切 2 个实例（M half）            | `standard_flow/phaseB_v8/gen_v8_onnx.py`                   | ~2×     |
| 2 | ONNX 里 pack_wt + MatMulV8 切 2 个实例（N half）  | 同上                                                       | ~1.5×   |
| 3 | 用 vshuffvdd 重写 `pack_act_rm_hvx.c`             | `kernel/pack_act_rm_hvx.c`                                 | ~3×     |
| 4 | 用 vshuffvdd 重写 `pack_wt_v3_hvx.c`              | `kernel/pack_wt_v3_hvx.c`                                  | ~2×     |
| 5 | 加 weights-to-VTCM prefetch op（HMX resource）    | 新 kernel + Interface 注册                                 | ~1.2×   |
| 6 | `TcmDramCopy` → `UntileToRowMajor`（不改图形状）  | `standard_flow/phaseB_v8/gen_v8_onnx.py`（换 op type）     | ~1.1×   |

步骤 1+2 只改 Python，不动 kernel。**先做这两步验证多节点拓扑能吃到
QNN scheduler 的 HVX 并行** —— 如果能，再投入 3+4 的 HVX kernel 改写。

## 8. 被证伪的假设（不要再做）

- ❌ "HMX 有 pipelining unlock"：`:dilate` / `mxswapacc` / `:retain` 都对
  `:cm:sat.ub` 不生效。Rt_wt=0x3FF 已经是硅 ceiling（7.9 cyc/packet）。
- ❌ "HMX 和 HVX 有并行 overlap"：实测 QNN 里 HMX 活跃的时候 HVX 只在做
  OutputSlice，**不并行**；HVX 活跃的时候 HMX 在 idle。重叠发生在
  HVX pack ↔ HMX weights_to_vtcm prefetch 这 9.6K cyc 小窗口。
- ❌ "多 HMX 线程"：只有一个 HMX unit，`tid=256` 唯一。串行。
- ❌ "row-major 输出 scatter 能优化":`mmv8` 里 sat.ub 到 row-major DDR
  的 32-row scatter ≈ 1.3M cyc at 512³，是硬 DDR 延迟。出路是 tile-layout
  output (V8 已经改)，不是优化 scatter。

## 9. 4096³ 数据：slicing 不是 perf 选项，是 **硬性要求**

4096³ w8a8，同样在 SM8650 v75 上。**这次 HMX 实际变成瓶颈了**。

### QNN native 在 4096³

| 指标                    | 数值              | 说明                                |
|-------------------------|-------------------:|-------------------------------------|
| timeline cycles         |     28,875,162    |                                     |
| graph_execute_us        |          14,897   | ~15 ms                              |
| **HMX utilization**     |           **95.6%** | vs 30% @ 512³ —— HMX 被完全压满   |
| HMX cycles used         |      27,597,579   |                                     |
| HVX cycles used (sum)   |       8,235,735   | 4 个 HVX 线程                       |
| inf/s                   |              65   | ≈ 4.47 TMAC/s                       |
| @Spill events           |  2020 × 17.9M cyc | **VTCM 溢出** → DDR staging        |
| @Fill events            |  1994 × 13.8M cyc |                                     |

### QNN 在 4096³ 的拓扑变了

| 节点               | @ 512³             | @ 4096³            |
|--------------------|-------------------:|-------------------:|
| `ConvLayer_s1.opt` |                 4 |           **2048** | M/256 × N/64 grid = 16×64, × 2 inferences |
| HVX threads (pack) |                 2 |              **4** | 512/513/514/515 全上 |
| `InputSlicePad`    |                 2 |                50 |                    |
| `ForceFormat_Crouton` |              2 |                42 |                    |
| `weights_to_vtcm`  |                 3 |            1,021  |                    |
| `bias_to_vtcm`     |                 2 |            1,060  |                    |
| `@Spill/@Fill`     |                 0 |           **4014** | **NEW** at scale   |
| `Concat`           |                 2 |               16  | 16 × [1,8,32,4096] |

每 ConvLayer 输出 **`[1, 8, 32, 64]`** = 256 × 64 tile（vs 512³ 的 256 × 256）。
就是说 QNN 在 4096³ 把 N 维切到 64 一块，M 维切到 256 一块 —— **tile 比 512³
更小**，因为要留 VTCM 容量给 spill/fill buffer。

每个 ConvLayer 平均 5,000 cyc（冷 call 14K，热 call 5K），`Cycles per Packet`
从冷的 16 降到热的 **5.71 cyc/packet**，已经打穿了 512³ 时观察到的 13.2
cyc/packet —— 大 shape 下 HMX 更接近硅 ceiling（7.89 理论下限）。

### V8 在 4096³ **跑不了**（ctxgen 阶段 fail）

```
[ ERROR ] "pack_act" requires 16777216 bytes (16 MB) > VTCM 8 MB
[ ERROR ] "pack_wt"  requires 33554432 bytes (32 MB)
[ ERROR ] "mmv8"     requires 50366464 bytes (50 MB)
[ ERROR ] "tcm2ddr"  requires 16777216 bytes (16 MB)
[ ERROR ] Graph prepare TCM Migration action failed
```

V8 的 `packed_act` tensor shape = `[1, M/32, K/32, 1024]` = `[1, 128, 128, 1024]` = **16 MB**，直接超过 VTCM 8 MB。类似地 `packed_wt` 16 MB，`mmv8`
工作集 50 MB。**qnn-context-binary-generator 在 graph prepare 就报错退出**，
连二进制都生成不了。

### 规模上的阈值

| Shape  | packed_act  | VTCM 能装 | V8 monolithic OK? |
|--------|------------:|----------:|-------------------|
| 512³   |   256 KB    | ✓         | yes               |
| 1024³  |     1 MB    | ✓         | yes               |
| 2048³  |     4 MB    | ✓         | 刚好              |
| 4096³  |    16 MB    | ✗         | **no，需要图切分**|
| 8192³  |    64 MB    | ✗✗        | no                |

QNN 用 **graph-level slicing + @Spill/@Fill** 来吞任意大 shape。V8 的 monolithic
图没这能力。

### 把这个转成 blueprint 更新

**Blueprint 第 7 节"实际动手顺序"的 #1 从"可选 3× 加速"升级为"必做"**，
否则 shape ≥ 2048 就开始撞 VTCM 墙。切法：

```
4096³ 建议切分（参考 QNN 的 16×64 grid）：
  M 切 16 份 (每份 256 行)
  N 切 64 份 (每份 64 列)   ← QNN 在 4096³ 切这么细是为了留 VTCM 给 spill
  K 不切（K=4096 整条串着扫）
  → 16 × 64 = 1024 个 MatMulV8 子节点
  → 16 个 pack_act 实例（每个 M_row_group）
  → 64 个 pack_wt 实例（每个 N_col_group）
  → Concat 层级：per-M-row-group concat N → 16 个 row stripes
             + 最外层 concat M → 最终输出
```

具体切到什么粒度要看 VTCM 预算：每 MatMulV8 实例的 working set ≤ 
VTCM - spill_buffer_reserved。参考 QNN 512³ 用 [1,8,32,256] tile (= 256×256 = 64K elements)，
4096³ 用 [1,8,32,64] (= 256×64 = 16K elements)。tile 粒度可由生成脚本参数化。

### trace 文件

- `trace_for_review/qnn_native_w8a8_4096.chrometrace.json` — 4096³ QNN 全量 trace（9.7 MB，很大）
- `trace_for_review/qnn_native_w8a8_4096.qhas_summary.html` — 资源汇总，浏览器直开

## 10. Cross-refs

- `Agent/qnn_hmx_pipelining.md` — HMX packet-level RE (Rt_wt, :cm 语义)
- `Agent/qnn_vs_v8_root_cause_2026-04-24.md` — 为什么 V8 用 tile-layout output
- `Agent/forceformat_crouton_re.md` — `convert_to_crouton_b/h` 反汇编 + vshuff 拓扑
- `Agent/hmx_sat_ub_semantics_2026-04-24.md` — `:after:cm:sat.ub` 像素级语义
- `Agent/v8_vs_native_optrace_2026-04-25.md` — 512³ 数据来源
- `docs/qnn_custom_op_sop.md` — 多节点 ONNX 怎么写
- `example/qnn_matmul_profile/sweep_data_4096/w8a8/` — 4096³ QNN 全部原始数据
- `trace_for_review/` — 汇总给 review 用的 trace 包（512³ + 4096³）
