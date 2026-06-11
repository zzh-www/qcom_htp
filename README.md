# qcom_htp

Qualcomm HTP / HMX MatMul kernel experiments, QNN Native reference flows, custom
op alignment work, and device-backed E2E CI evidence.

## ⭐ Highlight — GDN triangular inverse on HTP

The flagship kernel: a per-head **T = (I − A)⁻¹** solve for GDN / KDA linear-attention,
running HVX-feed + HMX-compute on the v75 HTP. **oc 3.10×10⁻³ (4.4× better than the int8
baseline) at ~2.2× the speed of the pure-HVX route.** Visual writeup — architecture, algorithm,
performance, and reproduce steps — in **[docs/gdn_inverse.md](docs/gdn_inverse.md)**.

Agent-managed project knowledge lives under [Agent/](Agent/README.md).  The
kernel CI source of truth is
[Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md) — it
holds the full per-case correctness matrices, the formal gate definition, and
the `ARTIFACT_ONLY=1` smoke-test notes.

## Correctness gates

Each promoted kernel family passes a same-hardware custom-vs-native output
exactness gate at `256x256x256`, `CHAIN=1`, across 7 float-derived cases
(`normal_random`, `zp_neutral`, `positive_boundary`, `negative_boundary`,
`single_k_impulse`, `bias_only`, `scale_only`).  "Promoted" means every case
reaches custom/native `65536/65536` with max integer delta `0` and a byte-exact
sidecar.

| Family | Correctness gate | Promoted status |
|---|---|---|
| `u8i8` | `correctness/test_u8i8_native_match_e2e.sh` | `normal_random`, output `65536/65536`, sidecar `2048/2048` |
| W4A8 per-channel | `correctness/test_w4a8_per_channel_native_match_e2e.sh` | 7/7 cases exact, sidecar `2048/2048` |
| W8A16 per-channel | `correctness/test_w8a16_per_channel_native_match_e2e.sh` | 7/7 cases exact, sidecar `4096/4096` |
| W4A16 per-channel | `correctness/test_w4a16_per_channel_native_match_e2e.sh` | 7/7 cases exact, sidecar `4096/4096` |
| W4A8 LPBQ | `correctness/test_w4a8_lpbq_native_match_e2e.sh` | 7/7 cases exact, sidecar `2048/2048` |
| W4A16 LPBQ | `correctness/test_w4a16_lpbq_native_match_e2e.sh` | 7/7 cases exact, sidecar `4096/4096` |
| **GDN inverse** | `correctness/test_gdn_solve_e2e.sh` | device `oc 3.10e-3 ≤ 1.05e-2` vs fp64 inv(I−A), C=256, 32 heads ([docs/gdn_inverse.md](docs/gdn_inverse.md)) |

The native/Python integer and dequant-float columns in the full matrix are
oracle tolerance checks (dequant uses `(q + qnn_offset) * output_scale`); the
promoted gate is the same-hardware custom/native exactness column.  The full
per-case matrices live in
[Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md) and are
regenerated, not stored — reproduce them with the runners below.

## Reproduce

Runners live under `tests/qnn_kernel_e2e/correctness/`.  Each rebuilds the QNN
op package, runs QNN Native first, syncs the custom case to the native DLC, then
checks same-hardware exactness on device.  Point `KERNEL_E2E_OUT_ROOT` at any
writable directory to collect evidence (it defaults to
`example/qnn_matmul_profile/`):

```bash
KERNEL_E2E_OUT_ROOT="$PWD/ci_evidence" \
  tests/qnn_kernel_e2e/correctness/test_w4a8_per_channel_native_match_e2e.sh
# per-case summary:
#   $KERNEL_E2E_OUT_ROOT/output_w4a8_per_channel_native_match_ci/analysis/custom_native_compare_summary.json
```

Swap the runner name and the matching `output_<family>_native_match_ci/`
directory for any other family in the table above.

## Full gate and push

```bash
tests/qnn_kernel_e2e/run_all.sh              # full device-backed gate (correctness then performance)
scripts/run_kernel_ci_preflight.sh           # run the gate, record a proof, then: git push
scripts/push_with_kernel_ci.sh origin main   # combined preflight + push
scripts/install_git_hooks.sh                 # enable the repo-local pre-push hook in this checkout
```

The pre-push hook does not re-run the long device gate while the GitHub SSH
connection is open.  It verifies the local proof written by
`scripts/run_kernel_ci_preflight.sh` and rejects the push unless the proof's
`HEAD` and tree match each pushed commit.  See
[Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md) for the
leaf-gate list, per-case matrices, and `ARTIFACT_ONLY=1` behavior.
