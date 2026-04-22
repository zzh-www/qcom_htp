# Simulator ↔ Device cycle parity (2026-04-22)

Phase 5 of the three-kernel plan. Goal: verify that `hexagon-sim --mv75
--mhmx 1` cycle counts match device silicon cycles within 5% for each of
w4a8 / w4a16 / w16a16, or document the divergence.

## Status

**Limited parity check completed.** The existing `tests/test_hmx_matmul_int16.sh`
simulator harness passes correctness for the single-tile int16 kernel
(`im_matmul_hmx_i8` in `example/hmx_matmul_int16/`). Three scenarios
bit-exact vs int64 reference; simulator Pcycles reported.

Full sim-vs-device cycle parity for the three QNN op-wrapped kernels
requires standalone sim harnesses (hexagon-sim binaries invoking the
HMX kernel functions outside the QNN framework). Those harnesses are
not yet written for w4a8 / w4a16 / w16a16 — each would need:
- An `int main()` that powers up HMX, allocates VTCM, fills test data,
  calls the kernel, and verifies vs a scalar reference
- A build.sh variant that produces a hexagon-sim ELF (not a .so)
- Wiring into `tests/test_hmx_matmul_<variant>.sh`

## Current device cycle baselines (for later comparison)

| kernel  | 32³      | 128³     | 512³      | source |
|---------|---------:|---------:|----------:|--------|
| w4a16   |   8.37   |   —      |   2.08    | `example/hmx_matmul_qnn/run_on_device.sh` |
| w4a8    |   5.74   |   2.65   |   1.92    | `example/hmx_matmul_w4a8/run_on_device.sh` |
| w16a16  |  20.78   |  12.37   |  12.23    | `example/hmx_matmul_w16a16/run_on_device.sh` |
| int16 single-tile (sim only) | 4.87M pcyc for 3 scenarios | | | `tests/test_hmx_matmul_int16.sh` |

## Expected divergence sources (when full parity is checked)

- **VTCM bank contention** — hexagon-sim's timing model may be
  optimistic about simultaneous VTCM accesses; silicon has bank-conflict
  stalls.
- **HMX pipeline latency** — sim likely models the HMX MAC as fixed
  latency; silicon has effective pipelining depth tied to `:cm` and Rt
  values (as documented in `Agent/qnn_hmx_pipelining.md`).
- **Memory hierarchy** — sim models a flat memory latency; device has
  L2 cache effects for DDR reads in `gather_w_col` etc.
- **HVX-thread parallelism** — sim single-threads by default; device
  can (for graph-level parallel ops) run on multiple HVX threads.

## Next steps (deferred)

1. Write standalone sim harnesses for each QNN-op kernel — each calls
   `hmx_int4xint8_matmul_mn` / `hmx_int4_matmul_mn_dualacc` /
   `hmx_int16x16_matmul_mn` directly with known inputs, measures pcycles.
2. Add build.sh variant producing `hexagon-sim` ELF.
3. Cross-compare with the corresponding `run_on_device.sh` cycle counts.

## Quick verification command

```sh
bash tests/test_hmx_matmul_int16.sh
# PASS: HMX int16 matmul E2E (3/3 scenarios bit-exact)
# (int16 single-tile sim kernel still green; establishes simulator
#  correctness as infrastructure baseline for future parity work)
```
