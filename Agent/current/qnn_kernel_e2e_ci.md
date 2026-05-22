# QNN Kernel E2E CI

This is the required CI gate for the canonical QNN HMX MatMul kernels and the
QNN-free handwritten HMX MatMul runtime:

| Group | Scope | CI entry |
|---|---|---|
| correctness | same-hardware exactness and CHAIN=1 precision | `tests/qnn_kernel_e2e/run_correctness.sh` |
| performance | per-channel canonical profile / CHAIN=8 E2E | `tests/qnn_kernel_e2e/run_performance.sh` |
| lpbq | lower-priority LPBQ performance smoke | `scripts/run_qnn_kernel_e2e_ci.sh lpbq` |

Leaf entries:

| Group | Kernel | CI entry | Promoted status |
|---|---|---|---|
| performance | `u8i8` | `tests/qnn_kernel_e2e/performance/test_u8i8_e2e.sh` | canonical CHAIN=8 custom/native profile |
| performance | `w4a8_per_channel` | `tests/qnn_kernel_e2e/performance/test_w4a8_per_channel_e2e.sh` | per-channel CHAIN=8 custom/native profile |
| performance | `w8a16` | `tests/qnn_kernel_e2e/performance/test_w8a16_e2e.sh` | canonical CHAIN=8 custom/native profile |
| performance | `w4a16_per_channel` | `tests/qnn_kernel_e2e/performance/test_w4a16_per_channel_e2e.sh` | per-channel CHAIN=8 custom/native profile |
| lpbq | `w4a8_lpbq` | `tests/qnn_kernel_e2e/performance/test_w4a8_lpbq_e2e.sh` | explicit lower-priority smoke |
| lpbq | `w4a16_lpbq` | `tests/qnn_kernel_e2e/performance/test_w4a16_lpbq_e2e.sh` | explicit lower-priority smoke |
| correctness | `u8i8_native_match` | `tests/qnn_kernel_e2e/correctness/test_u8i8_native_match_e2e.sh` | pass/promoted |
| correctness | `w4a8_per_channel_native_match` | `tests/qnn_kernel_e2e/correctness/test_w4a8_per_channel_native_match_e2e.sh` | pass/promoted |
| correctness | `w8a16_per_channel_native_match` | `tests/qnn_kernel_e2e/correctness/test_w8a16_per_channel_native_match_e2e.sh` | pass/promoted |
| correctness | `w4a16_per_channel_native_match` | `tests/qnn_kernel_e2e/correctness/test_w4a16_per_channel_native_match_e2e.sh` | pass/promoted |
| correctness | `w4a16_per_channel_chain1` | `tests/qnn_kernel_e2e/correctness/test_w4a16_per_channel_chain1_e2e.sh` | pass/promoted |
| correctness | `handwritten_hmx_matmul` | `tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh` | pass/promoted |

Machine-readable promoted correctness status is recorded in
`tests/qnn_kernel_e2e/correctness/status.json`.

Run the full gate before every formal code commit or push:

```bash
tests/qnn_kernel_e2e/run_all.sh
```

Install the repo-local pre-push hook in each checkout:

```bash
scripts/install_git_hooks.sh
```

The hook is versioned at `.githooks/pre-push` and invokes
`tests/qnn_kernel_e2e/run_all.sh` before Git sends refs to the remote.
The hook rejects `ARTIFACT_ONLY=1`; push-time validation must run on the device.

The full gate first runs correctness CI, then performance CI.  Correctness CI
prioritizes same-hardware exactness and explicitly runs only the promoted leaf
entries in `tests/qnn_kernel_e2e/run_correctness.sh`.  The `u8i8_native_match`
entry generates a
float-derived u8i8 Python/QNN case, runs QNN Native Conv1x1, builds the custom
HMX op with the recovered per-channel HTP prepare bias rule, and checks sidecar plus output
exactness against QNN Native.  Raw native sidecar injection is retained only as
a diagnostic override (`USE_NATIVE_BIAS_RECORD=1`).  Python/AIMET tolerance only
applies to the preceding QNN Native-vs-Python oracle check.  The default case is
`U8I8_MATCH_CASE=normal_random` at the validated canonical `256x256x256`;
smaller custom u8i8 shapes are not promoted by this gate.

The `w4a8_per_channel_native_match` entry is promoted separately from
`w4a8_lpbq`.  It generates the 7 float-derived per-channel cases
(`normal_random`, `zp_neutral`, `positive_boundary`, `negative_boundary`,
`single_k_impulse`, `bias_only`, `scale_only`) at `256x256x256`, runs QNN
Native first, syncs the custom static bias to QNN Native's quantized DLC `B`,
then checks both the generated W4A8 sidecar and same-hardware custom/native
output exactness.  Current evidence:
`/tmp/qcom_htp_w4a8_per_channel_native_match_ci/analysis/custom_native_compare_summary.json`
with every case `65536/65536`, maxabs `0`, and sidecar `2048/2048`.

The `w8a16_per_channel_native_match` entry is promoted.  It runs the 7
float-derived W8A16 cases (`normal_random`, `zp_neutral`,
`positive_boundary`, `negative_boundary`, `single_k_impulse`, `bias_only`,
`scale_only`) at `256x256x256`.  It runs QNN Native first, syncs the custom
case to the quantized DLC `B` and actual DLC A/Y/W encodings, builds the custom
A16 sidecar from the recovered generated drain/control and effective-bias
rules, then checks same-hardware custom/native output exactness.  Current
evidence:
`/tmp/qcom_htp_w8a16_ci_default_generated/output_w8a16_per_channel_native_match_ci/analysis/custom_native_compare_summary.json`
with every case `65536/65536`, maxabs `0`.  The recovered W8A16 drain scale
uses QNN's two-stage normalized scale path instead of the direct per-channel
`act_scale * weight_scale / output_scale` expression.  The prior `zp_neutral`
native-final-sidecar blocker and the rejected `HMX_W8A16_INTERNAL_SPLIT_N128`
hypothesis are recorded in `Agent/current/w8a16_zp_neutral_optrace.md`.  The
custom/native sidecar ABI difference and generated-sidecar implementation are
documented in `Agent/guides/qnn_htp_w8a16_perchannel_sidecar.md`.

The `w4a16_per_channel_native_match` entry is promoted.  It runs the same 7
float-derived Python/QNN cases at `256x256x256`, syncs the custom case to QNN
Native's quantized DLC `B` and actual DLC A/Y/W encodings, then builds a
generated A16 sidecar.  W4A16 uses the same normalized two-stage A16 drain
scale path as W8A16; treating the A16 control/drain words as fixed constants
only matched effective-bias fields and left the 512-byte sidecar wrong.
Current evidence:
`/tmp/qcom_htp_w4a16_per_channel_native_match_generated_ci2/output_w4a16_per_channel_native_match_ci/analysis/custom_native_compare_summary.json`
with every case sidecar `4096/4096`, control bytes `2048/2048`, effective
fields `256/256`, and same-hardware output `65536/65536`, maxabs `0`.

Performance CI is now limited to the per-channel canonical profile / CHAIN=8 E2E
flows: `u8i8`, `w4a8_per_channel`, `w8a16`, and `w4a16_per_channel`.  It rebuilds each QNN op package,
runs the canonical 256^3 device E2E flow, and validates the custom artifact
against the retained native reference.
It checks the repo-standard artifact layout, native I/O, NONTRIVIAL layout
flags, non-float runtime storage, bit-exact custom/native output, and optrace
evidence that the expected custom HMX op executed.
The `w4a16_per_channel_chain1` correctness entry is a precision case suite.  By default it
covers `default`, `zp`, `k_impulse0`, `k_impulse1`, and `k_impulse7`.  For each
case it regenerates a matched CHAIN=1 QNN-native Conv oracle before running the
custom op, then blocks on both custom/native exactness and
custom/Python-reference exactness.  This is the regression gate for the W4A16
floor256 accumulator-drain model.  Override the case list with
`W4A16_CHAIN1_CASES="default zp k2 k31"` when probing extra K lanes.
LPBQ entries are intentionally lower priority.  They are retained as explicit
targets under `scripts/run_qnn_kernel_e2e_ci.sh lpbq`; they are not part of the
default performance group while the active work is per-channel custom/native
alignment.  When run explicitly, the W4 LPBQ entries still verify that
`quant_overrides.json` uses QAIRT v1.0.0 LPBQ metadata for the `weight` tensor
(`compressed_bw=4`, `block_size=32`) so a symmetric W4 run cannot accidentally
satisfy the LPBQ CI case.
Use `ARTIFACT_ONLY=1` only as a cheap local smoke test when no device is
available:

```bash
ARTIFACT_ONLY=1 tests/qnn_kernel_e2e/run_all.sh
```

`ARTIFACT_ONLY=1` validates the retained standard artifacts under
`example/qnn_matmul_profile/`; it is not a substitute for the formal pre-push
device E2E gate after code changes, and the pre-push hook will fail if it is
set.
Because the CHAIN=1 W4A16 per-channel oracle is generated on demand,
`run_all.sh` skips `w4a16_per_channel_chain1` in `ARTIFACT_ONLY=1` mode unless
a retained chain1 artifact is promoted later.

The handwritten HMX MatMul correctness entry delegates to
`tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh`.  That sub-gate remains
the local implementation harness for direct-body custom-baseline checks; the
formal CI and pre-push entrypoint is the wrapper under `tests/qnn_kernel_e2e/`.
