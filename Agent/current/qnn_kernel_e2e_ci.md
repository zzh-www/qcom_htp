# QNN Kernel E2E CI

This is the required CI gate for the canonical QNN HMX MatMul kernels:

| Kernel | CI entry |
|---|---|
| `u8i8` | `tests/qnn_kernel_e2e/test_u8i8_e2e.sh` |
| `w4a8` | `tests/qnn_kernel_e2e/test_w4a8_e2e.sh` |
| `w4a8_lpbq` | `tests/qnn_kernel_e2e/test_w4a8_lpbq_e2e.sh` |
| `w8a16` | `tests/qnn_kernel_e2e/test_w8a16_e2e.sh` |
| `w4a16` | `tests/qnn_kernel_e2e/test_w4a16_e2e.sh` |
| `w4a16_chain1` | `tests/qnn_kernel_e2e/test_w4a16_chain1_e2e.sh` |
| `w4a16_lpbq` | `tests/qnn_kernel_e2e/test_w4a16_lpbq_e2e.sh` |
| `w16a16` | `tests/qnn_kernel_e2e/test_w16a16_e2e.sh` |

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

The full gate rebuilds each op package, runs the canonical 256^3 device E2E
flow, then validates the custom artifact against the retained native reference.
It checks the repo-standard artifact layout, native I/O, NONTRIVIAL layout
flags, non-float runtime storage, bit-exact custom/native output, and optrace
evidence that the expected custom HMX op executed.
The `w4a16_chain1` entry is a precision case suite.  By default it covers
`default`, `zp`, `k_impulse0`, `k_impulse1`, and `k_impulse7`.  For each case it
regenerates a matched CHAIN=1 QNN-native Conv oracle before running the custom
op, then blocks on both custom/native exactness and custom/Python-reference
exactness.  This is the regression gate for the W4A16 floor256
accumulator-drain model.  Override the case list with
`W4A16_CHAIN1_CASES="default zp k2 k31"` when probing extra K lanes.
For W4 LPBQ entries, it also verifies that `quant_overrides.json` uses QAIRT
v1.0.0 LPBQ metadata for the `weight` tensor (`compressed_bw=4`,
`block_size=32`) so a symmetric W4 run cannot accidentally satisfy the LPBQ CI
case.
For W16A16, cycle-budget analysis is recorded as diagnostics but is not a
push-blocking check because repeated device runs can move across the narrow
native-cycle threshold; output exactness, scoped accepted profile, justified
boundary policy, packet budget, and optrace execution remain blocking.

Use `ARTIFACT_ONLY=1` only as a cheap local smoke test when no device is
available:

```bash
ARTIFACT_ONLY=1 tests/qnn_kernel_e2e/run_all.sh
```

`ARTIFACT_ONLY=1` validates the retained standard artifacts under
`example/qnn_matmul_profile/`; it is not a substitute for the formal pre-push
device E2E gate after code changes, and the pre-push hook will fail if it is
set.
Because the CHAIN=1 W4A16 oracle is generated on demand, `run_all.sh` skips
`w4a16_chain1` in `ARTIFACT_ONLY=1` mode unless a retained chain1 artifact is
promoted later.

The W16A16 gate is intentionally scoped to the accepted canonical native-record
contract: `W16A16_KERNEL_PROFILE=accepted`, `CHAIN=1`, `MODE=chain_qdq`, and
native oracle `output_w16a16_native_ref_e2e_256`.
