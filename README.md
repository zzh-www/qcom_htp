# qcom_htp

Qualcomm HTP / HMX MatMul kernel experiments, QNN Native reference flows, custom
op alignment work, and device-backed E2E CI evidence.

Agent-managed project knowledge lives under [Agent/](Agent/README.md).  The
current kernel CI source of truth is
[Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md).

## Current CI Correctness Matrix

This table mirrors the CI archive in
[Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md).  Keep
both documents in sync when refreshing device results.

All rows below are same-shape `256x256x256`, `CHAIN=1`, 7-case device results
unless otherwise noted.  The native/Python integer columns are oracle tolerance
checks.  The dequant float columns use final output dequantization:

```text
(q + qnn_offset) * output_scale
```

The promoted gate is same-hardware custom/native output exactness.
The full CI gate `tests/qnn_kernel_e2e/run_all.sh` now includes the promoted
W4A8/W4A16 LPBQ native-match correctness tests through
`tests/qnn_kernel_e2e/run_correctness.sh`.

The `u8i8` promoted gate is a single `normal_random` case:
custom/native `65536/65536`, max integer delta `0`, sidecar `2048/2048`.

### W4A8 per-channel

Source evidence: `/tmp/qcom_htp_display_perchannel_w4a8`.

| case | Native vs Python exact ints | Max int delta | Mean int delta | Max dequant float delta | Mean dequant float delta | Custom vs native exact | Custom vs native max int delta | Sidecar bytes | Sidecar control | Sidecar effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_random` | 64761/65536 | 1 | 0.01183 | 0.058093 | 0.000687 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `zp_neutral` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `positive_boundary` | 63826/65536 | 1 | 0.02609 | 0.503116 | 0.013128 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `negative_boundary` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `single_k_impulse` | 64000/65536 | 1 | 0.02344 | 0.000623 | 0.000015 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `bias_only` | 64796/65536 | 1 | 0.01129 | 0.003399 | 0.000038 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `scale_only` | 64788/65536 | 1 | 0.01141 | 0.477660 | 0.005452 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |

### W8A16 per-channel

Source evidence: `/tmp/qcom_htp_display_perchannel_w8a16`.

| case | Native vs Python exact ints | Max int delta | Mean int delta | Max dequant float delta | Mean dequant float delta | Custom vs native exact | Custom vs native max int delta | Sidecar bytes | Sidecar control | Sidecar effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_random` | 61666/65536 | 1 | 0.05905 | 0.000726 | 0.000043 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `zp_neutral` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 4095/4096 | 2048/2048 | 255/256 |
| `positive_boundary` | 64987/65536 | 1 | 0.00838 | 0.006156 | 0.000052 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `negative_boundary` | 65231/65536 | 1 | 0.00465 | 0.006152 | 0.000029 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `single_k_impulse` | 33024/65536 | 1 | 0.49609 | 0.000008 | 0.000000 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `bias_only` | 64665/65536 | 1 | 0.01329 | 0.000028 | 0.000000 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `scale_only` | 62976/65536 | 1 | 0.03906 | 0.005532 | 0.000216 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |

### W4A16 per-channel

Source evidence: `/tmp/qcom_htp_display_perchannel_w4a16`.

| case | Native vs Python exact ints | Max int delta | Mean int delta | Max dequant float delta | Mean dequant float delta | Custom vs native exact | Custom vs native max int delta | Sidecar bytes | Sidecar control | Sidecar effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_random` | 18182/65536 | 3 | 1.03569 | 0.001710 | 0.000590 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `zp_neutral` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `positive_boundary` | 53853/65536 | 1 | 0.17827 | 0.004648 | 0.000829 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `negative_boundary` | 61743/65536 | 1 | 0.05788 | 0.004619 | 0.000267 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `single_k_impulse` | 18688/65536 | 35 | 9.32422 | 0.000199 | 0.000053 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `bias_only` | 50490/65536 | 1 | 0.22958 | 0.000026 | 0.000006 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `scale_only` | 20527/65536 | 2 | 0.80811 | 0.007646 | 0.003089 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |

### W4A8 LPBQ

Source evidence: `/tmp/qcom_htp_lpbq_w4a8_full_pergroup`.

| case | Native vs Python exact ints | Max int delta | Mean int delta | Max dequant float delta | Mean dequant float delta | Custom vs native exact | Custom vs native max int delta | Sidecar bytes | Sidecar control | Sidecar effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_random` | 64705/65536 | 1 | 0.01268 | 0.058170 | 0.000738 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `zp_neutral` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `positive_boundary` | 63581/65536 | 1 | 0.02983 | 0.506812 | 0.015119 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `negative_boundary` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `single_k_impulse` | 64768/65536 | 1 | 0.01172 | 0.000625 | 0.000007 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `bias_only` | 64732/65536 | 1 | 0.01227 | 0.003345 | 0.000041 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |
| `scale_only` | 64728/65536 | 1 | 0.01233 | 0.403659 | 0.004977 | 65536/65536 | 0 | 2048/2048 | 1024/1024 | 1024/1024 |

### W4A16 LPBQ

Source evidence: `/tmp/qcom_htp_lpbq_w4a16_full_pergroup`.

| case | Native vs Python exact ints | Max int delta | Mean int delta | Max dequant float delta | Mean dequant float delta | Custom vs native exact | Custom vs native max int delta | Sidecar bytes | Sidecar control | Sidecar effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_random` | 20881/65536 | 3 | 0.93990 | 0.001591 | 0.000498 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `zp_neutral` | 65536/65536 | 0 | 0.00000 | 0.000000 | 0.000000 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `positive_boundary` | 54051/65536 | 1 | 0.17525 | 0.004621 | 0.000810 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `negative_boundary` | 61773/65536 | 1 | 0.05742 | 0.004608 | 0.000265 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `single_k_impulse` | 30720/65536 | 2 | 0.54688 | 0.000011 | 0.000000 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `bias_only` | 52997/65536 | 1 | 0.19133 | 0.000027 | 0.000005 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
| `scale_only` | 22455/65536 | 2 | 0.73369 | 0.007855 | 0.002882 | 65536/65536 | 0 | 4096/4096 | 2048/2048 | 256/256 |
