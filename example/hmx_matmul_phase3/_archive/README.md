# _archive/

Old V2-V7 custom-op code, probes, and simulator harnesses. **Not built** by
`build.sh` / `build_x86.sh` anymore; kept for reference (reverse-engineering
notes, int4 experiments, early HMX semantics probes).

After the 2026-04-25 V8 → ONNX → DLC → ctx-binary flow validation, only the V8
path is kept under active maintenance. See `docs/qnn_custom_op_sop.md`.

## Layout

| Dir                     | Contents                                                      |
|-------------------------|---------------------------------------------------------------|
| `src/`                  | V2/V3/V4/V6/V7 op kernels, RequantHvx, Phase3A probe, old host harnesses |
| `kernel/`               | Kernels for non-V8 paths (pack_act v1/v2, pack_wt v1, hmx_core{,_v2}, int4, combine_hi_lo) |
| `run_scripts/`          | `run_v{2..7}_*_on_device.sh`, old `run_graph_on_device.sh`, `build_sim_core.sh` |
| `test_core_sim.c`, `test_ops_sim.c` | Simulator-side unit tests for the old HMX core |

## Re-enabling any of these

1. Move the needed src / kernel / script back out of `_archive/`.
2. Add its compile unit to `build.sh` (search for `V8_SRCS`/`KERNEL_SRCS`).
3. Add its op-name + `register_*_op()` call to `src/HmxMatMulPhase3Interface.cpp`
   (`sg_opNames[]` and `qhpi_init()`).
4. Rebuild + push to device.

## Git history

Full history is in `git log -- example/hmx_matmul_phase3/` — the moves were
done via `git mv`, so blame/log chains for archived files are preserved.
