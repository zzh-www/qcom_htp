# QNN HMX MatMul Status

## Current Direction

The handwritten MatMul route is QNN-free at runtime.  The implementation lives
under `example/handwritten_hmx_matmul/` and is validated by:

```bash
OUT_ROOT=/tmp/handwritten_hmx_matmul_gate_device_refresh2 \
  DEVICE=oneplus tests/handwritten_hmx_matmul/run_all.sh
```

Current gate status: pass.

The gate promotes all active families:

- `u8i8`
- `w4a8`
- `w8a16`
- `w4a16`

W16A16 remains retained reference material only.

## W4A16 Acceptance

W4A16 is accepted through the QNN custom-op baseline, not by re-entering the
QNN native Conv graph.

Current evidence:

- `device_body_w4a16_chain8_custom_baseline.json`: direct-HMX device body is
  `byte_exact_device_diff`, byte diffs `0`, checksum `0xfcdb7a52`.
- `w4a16_chain8_custom_baseline_native_bridge.json`: custom/native public
  layout bridge is `accepted_bridge=true`.
- The required public-layout transform is the existing Python
  `native_transpose_2d` rule.
- `native_transpose_2d` is `65536/65536` exact with byte diffs `0`.

The old tutorial chain1 run remains a route-gate/provenance check.  It proves
prepared-state, call-ABI, VTCM-offset, step-trace, and HNH-path visibility, but
it is not the W4A16 acceptance oracle.

## Gate Artifacts

The last full device gate wrote:

- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/promotion_evidence.json`
- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/completion_checklist.json`
- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/roadmap_audit.json`
- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/m4_blockers.json`

Important machine-readable status:

- `promotion_evidence.json`: `m4_promoted_count=4`,
  `m4_promotable_count=4`, `unpromoted_families=[]`.
- `completion_checklist.json`: `roadmap_complete=true`, `pass=4`, `open=0`,
  `fail=0`.
- `roadmap_audit.json`: `w4a16_complete=true`, `pass=5`, `open=0`, `fail=0`.

## Historical Boundary

The old W4A16 QNN blackbox/native-entry investigation is closed for the current
implementation goal.  Its useful result is provenance: QNN Native/custom-op
artifacts define prepared bytes, raw output oracles, body slices, packet counts,
and native performance references.  They are not runtime dependencies.

Do not wire old descdump, ctxgen, descriptor mutation, selector sweep, or
residual/cvt microprobe matrices into the current gate.
