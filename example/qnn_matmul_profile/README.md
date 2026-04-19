# qnn_matmul_profile — end-to-end QNN MatMul cycle profiler

Runs a 32×32×32 MatMul through QNN HTP on a real SM8650 device for each
dtype combination, extracts per-op cycle counts via `optrace`, and prints
a summary table.

## Prerequisites

- `.venv/` with `uv sync` complete (Python 3.10, onnx, numpy, pyyaml).
- `scripts/env.sh` sourced (puts `hexagon-clang`, `qairt-converter`,
  `qnn-net-run`, and the venv on `PATH`).
- `ssh oneplus` resolves to a Snapdragon 8 Gen 3 (SM8650) device running
  a Termux shell.
- `~/qnn_run/` on the device populated with the QNN runtime (one-time):
  ```
  libQnnHtp.so              (from tools/qnn-sdk/lib/aarch64-android)
  libQnnSystem.so           (from tools/qnn-sdk/lib/aarch64-android)
  libQnnModelDlc.so         (from tools/qnn-sdk/lib/aarch64-android)
  libQnnHtpNetRunExtensions.so  (from tools/qnn-sdk/lib/aarch64-android)
  libQnnHtpV75Skel.so       (from tools/qnn-sdk/lib/hexagon-v75/unsigned)
  libQnnHtpV75Stub.so       (from tools/qnn-sdk/lib/aarch64-android)
  libQnnHtpV75.so           (from tools/qnn-sdk/lib/hexagon-v75/unsigned)
  qnn-net-run               (from tools/qnn-sdk/bin/aarch64-android)
  ```

## Usage

```sh
source scripts/env.sh
bash example/qnn_matmul_profile/profile_all.sh [flags]
```

Flags (each has an env-var fallback in parens):

| Flag             | Values                      | Default          | Env       |
|------------------|-----------------------------|------------------|-----------|
| `--device -d`    | ssh host alias / adb serial | `oneplus`        | `DEVICE`  |
| `--connect -c`   | `ssh` \| `adb`              | `ssh`            | `CONNECT` |
| `--arch -a`      | `v66`/`v68`/`v69`/`v73`/`v75`/`v79`/`v81` | `v75` | `ARCH` |
| `--out-dir`      | host artifact dir           | `$PWD/output`    | `OUT_DIR` |
| `--configs`      | space-separated dtype set   | all 7            | `CONFIGS` |
| `--shape`        | matmul dims `M,K,N`          | `32,32,32`       | `SHAPE`   |

Other env-only knobs:
- `NUM_INFERENCES=20`  inferences per `qnn-net-run` invocation
- `DEVICE_DIR`         remote workdir (defaults: ssh→`~/qnn_run`, adb→`/data/local/tmp/qnn_run`)
- `SOC_ID`             override HTP soc_id if the arch→id default is wrong for your chip

Examples:

```sh
# Default — SM8650 (v75) via ssh oneplus, 32×32×32 matmul
bash example/qnn_matmul_profile/profile_all.sh

# Larger 512×512×512 matmul (where dtype differences are visible)
bash example/qnn_matmul_profile/profile_all.sh --shape 512,512,512

# SM8750 (v79) over adb
bash example/qnn_matmul_profile/profile_all.sh --connect adb --device R9WW --arch v79

# Only fp16 + w8a8 on oneplus (faster debug)
bash example/qnn_matmul_profile/profile_all.sh --configs "fp16 w8a8"

# Size sweep across 32 / 128 / 256 / 512 (one run each)
bash example/qnn_matmul_profile/bench_sweep.sh 32 128 256 512 -- --configs "fp16 w16a16 w8a16 w8a8"

# Multi-run stability at fixed size
bash example/qnn_matmul_profile/bench_repeat.sh 5 -- --shape 512,512,512 --configs "fp16 w8a8"
```

Artifacts land under `$OUT_DIR/<config>/` (default `$PWD/output/<config>/`):
- `matmul.onnx` — generated source graph
- `quant_overrides.json` — activation/weight/output encodings
- `matmul.dlc` — converted QNN graph
- `schematic.bin` — topology for chrometrace mapping
- `profile.log` — raw optrace from device
- `chrometrace.json` — per-op cycle breakdown

`$OUT_DIR/summary.json` consolidates everything programmatically.

### Environment overrides

- `CONFIGS=...`     space-separated list (default: fp16 w16a16 w8a16 w8a8 w4a16 w4a8 w4a4)
- `NUM_INFERENCES=20`   iterations per run (amortizes one-shot noise)
- `OUT_DIR=...`     artifact root (default `/tmp/qnn_profile`)
- `SSH_HOST=...`    device alias (default `oneplus`)
- `DEVICE_DIR=...`  remote workdir (default `~/qnn_run`)

## Supported / unsupported configs

HTP v75 MatMul-op supported dtypes (from
`qti/aisw/converters/common/backend_aware_configs/htp_v2.json`), with the
HTP compile pipeline getting the correct `soc_id` (QNN SocModel enum, not
the Android hw revision id):

| activation × weight | observed HTP kernel (optrace `main_type`) |
|--|--|
| fp16 × fp16 | `q::ConvLayer.opt.weights_to_vtcm` |
| int16 × int16 | `q::ConvLayer.opt.bias_to_vtcm` (native int16, NOT fp16 emul) |
| int16 × int8  | `q::ConvLayer_s1.opt` |
| int8  × int8  | `q::ConvLayer.opt.weights_to_vtcm` |
| fp16  × int8  | (weight-only quant, not exercised here) |

> **Warning about soc_id**: the Android hw revision id
> (`/sys/devices/soc0/soc_id`, e.g. 577 for SM8650) is **NOT** the same as
> QNN's `socModel` enum (e.g. 57 for SM8650, from
> `tools/qnn-sdk/include/QNN/QnnTypes.h`). Feeding the wrong value makes
> ctxgen silently fall back to default (v68) and runtime lowers int16
> kernels to fp16 emulation (`q::ConvLayer.fp16.s1.tcm`), leading to
> misleading conclusions about "no native int16 support". The script's
> `ARCH_SOCID` table uses the right QNN enum values.

**Unsupported at QNN MatMul-op level:**
- `w4a16`, `w4a8`: HMX has a 4-bit weight path (`weight.n`) but QNN
  exposes it only via Conv2D+LPBQ, not MatMul. Returns
  `QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM` at compose time.
- `w4a4`: rejected by the quantizer with
  `Activation bitwidth conversion from 8 to 4 is not supported`.

## Interpreting the output table

Per `matmul_1` event the parser splits cycles into three buckets:

- **compute** — the actual HMX MAC kernel. Expected labels:
  - `q::ConvLayer.fp16.s1.tcm` for fp16 MAC
  - `q::ConvLayer_s1.opt` for any int-quantized MAC (int8×int8,
    int16×int16, int16×int8)
  - Seeing `q::ConvLayer.fp16.s1.tcm` on an int16 config means the
    HTP backend fell back to fp16 emulation — **almost always a wrong
    `soc_id` in `_htp_ext.json`** (see ARCH_SOCID table in the script).
- **staging** — `q::ConvLayer.opt.weights_to_vtcm` + `bias_to_vtcm`.
  One-time cost of laying out weight and encoding data into HMX tile
  format inside VTCM. Amortizable across repeated inferences with the
  same weights.
- **dma** — `DmaCheckpointSet`, `SyncOp`, `ChunkPreload` plumbing.

`input` / `output` columns are separate graph Input/Output ops (tensor
layout conversions on the framework's side of the graph).

## Known noise + recommended usage

**chrometrace.json only captures the FIRST inference** (the warmup run,
which runs with cold caches). The raw `profile.log` contains all 20
(or however many `--num_inferences`) iteration timings; use
`parse_profile_log.py` to read those and take the steady-state median.

Warmup-vs-steady ratio can be up to 6× at small sizes and ~1.3× at
large sizes. For dtype comparisons ALWAYS use steady-state numbers.

`bench_repeat.sh` runs the full profile N times which ALSO amortizes
noise, but is slower than `parse_profile_log.py` on a single 20-iter
run.

### Bimodal behavior

On SM8650 v75 some configs (int8 × int8, int8 × int16) at larger sizes
exhibit bimodal distribution — half the iterations are ~80k cycles and
half are ~140k. Unknown root cause (DCVS transitions, VTCM banking,
scheduler). Report both `steady` and `min` from `parse_profile_log.py`
and note which one you quote. `fp16` and `w16a16` do NOT show this.

Example (5 runs, skip unsupported w4* for speed):

```sh
bash bench_repeat.sh 5 -- --configs "fp16 w16a16 w8a16 w8a8"
```

Outputs per-metric `median [min-max]` + `agg_summary.json`.

A saved reference run (2026-04-19, SM8650 v75) lives under
`bench_data_2026-04-19/`. Headline numbers and analysis in
`Agent/qnn_matmul_dtype_comparison.md`.

## Files

```
gen_onnx.py            — per-config ONNX + quant_overrides + input emitter (M/K/N configurable)
profile_all.sh         — E2E driver (convert → ctxgen → device run → parse)
parse_qhas.py          — **recommended**. Read chrometrace_qnn_htp_analysis_summary.json → timeline_cycles, graph_execute_us, HMX util%, HMX/HVX cycles per config. **Fixes the QNN profiler-reader UNK bug** by (a) mapping TID 256→HMX / 512-515→HVX when htp_resources says `type=UNK`, (b) rebuilding I/O counters from chrometrace_htp.json when QHAS reports 0. Prints `fixup=T/I` when recovery kicked in.
parse_chrometrace.py   — aggregate chrometrace (first-inference matmul_1 event breakdown) — partial metric, use QHAS instead for dtype comparisons
parse_profile_log.py   — extract matmul_1:OpId cycles across all N iterations — useful for per-dtype scaling analysis only
bench_repeat.sh        — run profile_all N times at fixed size
bench_sweep.sh         — run profile_all across a sweep of matmul sizes
bench_data_2026-04-19/ — 5-run reference data at 32×32×32
sweep_data_2026-04-19/ — size sweep (32/128/256/512)
```

## Interpreting `SystemService(lumped)` in compute kernel

At small matmul sizes the optrace reader can demux each matmul sub-event
into a specific `q::...` kernel name. At larger sizes (≥256-ish) the
reader collapses everything under the matmul op into unnamed
`SystemService` events, so the `compute kernel` column shows
`SystemService(lumped)`. The **matmul** total column is still accurate.
To recover the real kernel names, read the `_htp.json` artifact next to
the `chrometrace.json` — it always has the compiled kernel node list.
