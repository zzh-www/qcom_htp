# QNN HMX MatMul Status

This page is the concise current-state source for MatMul kernel families.  Use
the linked guide and handoff documents for investigation history.

## Current Direction

The promoted QNN custom/native baseline now covers:

- per-channel `u8i8`, `w4a8`, `w8a16`, and `w4a16` correctness;
- W4A8/W4A16 LPBQ real-native correctness;
- W4A16 CHAIN=1 precision cases;
- QNN-free handwritten HMX MatMul correctness.

The formal full gate is:

```bash
tests/qnn_kernel_e2e/run_all.sh
```

The latest recorded full device-backed run for this document set passed on
2026-05-30 with `tests/qnn_kernel_e2e/run_all.sh`.  The pre-push hook also
runs this gate and rejects `ARTIFACT_ONLY=1`.

Machine-readable promoted correctness state lives in
`tests/qnn_kernel_e2e/correctness/status.json`.  Human-readable CI details and
the current comparison matrices live in
[qnn_kernel_e2e_ci.md](qnn_kernel_e2e_ci.md) and the repo
[README](../../README.md).

## QNN Custom/Native Kernels

Same-hardware comparisons remain exact-output gates.  Native-vs-Python oracle
tolerance is diagnostic only and does not relax custom/native or
handwritten/custom checks.

### Per-Channel Baseline

The accepted per-channel baseline is protected by default correctness and
performance CI:

| Target | Gate |
|---|---|
| `u8i8_native_match` | custom HMX vs QNN Native, single `normal_random` generated case |
| `w4a8_per_channel_native_match` | 7 generated per-channel cases |
| `w8a16_per_channel_native_match` | 7 generated per-channel cases |
| `w4a16_per_channel_native_match` | 7 generated per-channel cases |
| `w4a16_per_channel_chain1` | CHAIN=1 precision suite |

The recovered per-channel bias/drain behavior is documented in:

- [QNN HTP per-channel bias prepare](../guides/qnn_htp_perchannel_bias_prepare.md)
- [QNN HTP W8A16 per-channel sidecar](../guides/qnn_htp_w8a16_perchannel_sidecar.md)

### LPBQ

LPBQ correctness now compares against real QNN Native LPBQ Conv, not the older
per-channel Conv oracle.

Generated LPBQ cases use dedicated `w4a8_lpbq` and `w4a16_lpbq` families.  The
generator first derives signed int4 per-group scales on K32 blocks, then
factorizes them as:

```text
per_channel_float_scale * per_block_int_scale
```

The native runner validates that QNN lowers through:

```text
Conv2d_w_blk_exp_scale -> q::ConvLayer.opt.expand_block_quant_to_pc_int8_weights
```

The custom implementation mirrors that split:

```text
HmxW4LpbqExpandToI8 -> HmxU8I8ToU8MatMul
HmxW4LpbqExpandToI8 -> HmxU16I8ToU16MatMul
```

`HmxW4LpbqExpandToI8` expands native K-pair signed int4 weights plus per-block
integer scales into the K-major W8 HMX carrier.  Static LPBQ weights are
const-folded during prepare, so QNN inserts `weights_to_vtcm` before the W8
compute op.

Current promoted evidence:

| Target | Evidence | Result |
|---|---|---|
| `w4a8_lpbq_native_match` | `/tmp/qcom_htp_lpbq_w4a8_full_pergroup` | 7 cases, custom/native `65536/65536`, max integer delta `0` |
| `w4a16_lpbq_native_match` | `/tmp/qcom_htp_lpbq_w4a16_full_pergroup` | 7 cases, custom/native `65536/65536`, max integer delta `0` |

The older `/tmp/qcom_htp_lpbq_full_ci/` evidence used degenerate all-one LPBQ
metadata and must not be used as proof of real blockwise expansion support.

## Handwritten HMX MatMul

The handwritten MatMul route is QNN-free at runtime.  It prepares owned buffers,
calls replicated HMX bodies directly, and compares against retained offline
oracles.

Implementation and gate:

- `example/handwritten_hmx_matmul/`
- `tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh`
- `tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh`

The active gate promotes:

- `u8i8`
- `w4a8`
- `w8a16`
- `w4a16`

W16A16 remains retained reference material only.

W4A16 handwritten acceptance uses a retained QNN custom-op artifact as the
direct-HMX custom baseline, then checks the public custom/native bridge through
the existing `native_transpose_2d` transform.  The old tutorial chain1 wrapper
and W4A16 QNN blackbox route are closed as implementation paths.

Current handwritten details:

- [Handwritten HMX MatMul roadmap](../guides/handwritten_hmx_matmul_roadmap.md)
- [Handwritten HMX MatMul runtime boundary](handwritten_hmx_matmul_runtime.md)
- [Handwritten HMX MatMul body evidence](handwritten_hmx_matmul_bodies.md)
- [Handwritten HMX MatMul oracles](handwritten_hmx_matmul_oracles.md)

## Historical Boundaries

Use these boundaries when interpreting old notes:

- W4A16 QNN Native/custom-op blackbox work is historical provenance, not the
  active handwritten implementation route.
- W16A16 is not part of the active correctness or performance gates.
- Broad descdump, ctxgen, descriptor mutation, selector sweep, and residual/CVT
  microprobe matrices should not be wired into the current gates.
- QNN Native/custom-op artifacts may still define prepared bytes, raw output
  oracles, body slices, packet counts, and performance references.

Historical details are intentionally kept out of this status page:

- [w4a16 QNN native path](../handoffs/w4a16_qnn_native_path.md)
- [w4a16 native alignment handoff](../handoffs/w4a16_native_alignment.md)
- [w8a16 native alignment handoff](../handoffs/w8a16_native_alignment.md)
- [w16a16 native alignment plan](../handoffs/w16a16_native_alignment_plan.md)
- [QNN native alignment blackbox handbook](../guides/qnn_native_alignment_blackbox_handbook.md)
