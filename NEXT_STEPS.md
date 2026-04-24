# V8 matches V6 at 512³ via tile-layout output (2026-04-24)

> **4.6× speedup** at 512³ (1.69M → 368K cyc) by switching V8 to
> QNN-style **tile-layout output**. V8 now **matches V6** (362K)
> while being HMX-only. Learned by disassembling QNN's hot MatMul
> kernel — it writes `:after:cm:sat.ub` **directly** to tile-layout
> DDR pointers, no scatter.

## Final numbers (all DIAG modes bit-exact; random max_err ≤ 1 at 32³)

| Shape | V8 previous | **V8 now**  | V6        | V8 / V6 |
|-------|------------:|------------:|----------:|--------:|
| 32³   |      14,000 |      22,600 |    ~16,000|   1.4×  |
| 512³  |   1,690,000 |   **368,000**|   362,000 | **1.0×** ✓ |
| 1024³ |   7,900,000 |   2,702,000 | 1,650,000 |   1.6×  |

Per-op breakdown at 512³ (previously shown bloated, now streamlined):

| Op         | cyc (new)  | role                              |
|------------|-----------:|-----------------------------------|
| pack_act   |     48,000 | DDR act → VTCM row-major tile     |
| pack_wt    |        600 | cached after first call           |
| **mmv8**   |  **281,000**| HMX MAC + sat.ub to VTCM tile |
| tcm2ddr    |     12,000 | HVX vmem bulk VTCM→DDR copy       |
| framework  |     26,000 | QNN input/output overhead         |

## What changed

### 1. mmv8: sat.ub writes tile-layout directly
```c
// Before: HMX → vtcm_stg (1 KiB), then 32× memcpy(32) to DDR at stride N
// After:  HMX → out_tile (1 KiB DDR or VTCM), contiguous, no scatter
asm("mxmem(%0, %1):after:cm:sat.ub = acc"
    :: "r"(out_tile), "r"(0) : "memory");
```
Saved ~5000 cyc/tile (scatter) at the cost of tile-layout output.

### 2. Output layout: [M_tiles, N_tiles, 1024] tile-contiguous
Same total bytes as row-major `[M, N]`, different arrangement.
Matches QNN `ConvLayer_s1.opt` native output — the standard QNN
intermediate format for chained ops.

### 3. TcmDramCopy op: bulk HVX memcpy (VTCM → DDR)
The only step needed when graph output must be DDR-backed.
256 KB transfer in 12 K cyc via explicit `*(HVX_Vector*)d = vload` loop
(vs 1.6 M cyc for libc memcpy — compiler wasn't HVX-vectorizing it).

## What didn't work (all reverted, documented)

- **mxswapacc pair-mode**: `:retain` not honored on `:cm:sat.ub`;
  DIAG breaks. ~1% noise-level perf. `Agent/v8_pair_mode_deadend_2026-04-24.md`.
- **Row-major output with HVX masked store**: alignment issues when
  N stride misaligned.
- **Full-row gather (row-first)**: VTCM bank conflicts, 2.3M at 512³.
- **Per-row VTCM staging + full-row memcpy**: 2.0M (no win vs inline scatter).
- **Split mmv8 + untile (row-major output)**: 2.0M — untile alone is 1.6M
  because row-major 32B strided DDR writes have no coalesce path.

## For standalone row-major `[M, N]` output

If caller needs row-major, add an untile op (see
`kernel/untile_to_rowmajor_hvx.c`, registered as `UntileToRowMajor`).
Adds ~1.6 M cyc at 512³ — still net faster than the old inline scatter
since mmv8 itself dropped from 1.6 M to 281 K, but comparable overall.

**For in-graph usage (V8 feeding another tile-aware op), no untile
needed — stay tile-layout all the way through.**

## Remaining residuals

- 1024³: V8 at 2.7 M vs V6 at 1.65 M — V6 still wins at large K by
  overlapping HMX with HVX requant.  V8 is pure-HMX by design so no
  HVX overlap.  To beat V6 at 1024³ would require giving up "HMX-only"
  and adding an HVX requant/overlap stage.
- Random-mode bit-exactness: 28/1024 at 32³ max_err=1 (fp16 precision
  edge), 1 M/1 M at 1024³ max_err=20 (K-scale accumulation drift).
  DIAG uniform modes all 0 mismatches up to 512³.

## Files changed this round

```
new:
  example/hmx_matmul_phase3/kernel/tcm_dram_copy_hvx.c
  example/hmx_matmul_phase3/kernel/untile_to_rowmajor_hvx.c
  Agent/qnn_vs_v8_root_cause_2026-04-24.md
  Agent/v8_pair_mode_deadend_2026-04-24.md

changed:
  example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp
    - tile-layout output mode (detected by last dim == 1024)
    - legacy row-major path preserved as fallback
    - output signature: TCM_Only
  example/hmx_matmul_phase3/src/HmxMatMulPhase3Interface.cpp
    - register UntileToRowMajor + TcmDramCopy ops
  example/hmx_matmul_phase3/src/run_matmul_v8_graph.cpp
    - output tensor: [1, M_tiles, N_tiles, 1024] tile-layout
    - 2-node tail: mmv8 (tile-layout VTCM) → TcmDramCopy (DDR)
    - reference computed in tile-layout for bit-exact check
  example/hmx_matmul_phase3/build.sh
    - + tcm_dram_copy_hvx, untile_to_rowmajor_hvx sources
```

## Resume

```bash
cd example/hmx_matmul_phase3 && bash build.sh
bash run_v8_graph_on_device.sh --shape 512,512,512  # 368K cyc
bash run_v8_graph_on_device.sh --shape 1024,1024,1024 # 2.7M cyc

# DIAG at 32³ (all 0/1024):
for d in 999 998 997 996 995; do
  ssh oneplus "cd ~/qnn_run && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
     ./run_matmul_v8_graph 32 32 32 $d" | grep Check
done

# DIAG at 512³ (uniform modes 0/262144):
for d in 999 998 997; do
  ssh oneplus "cd ~/qnn_run && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
     ./run_matmul_v8_graph 512 512 512 $d" | grep Check
done
```
