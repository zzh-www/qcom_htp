# Handwritten HMX MatMul Oracles

This is the Milestone-0 oracle freeze for the QNN-free handwritten HMX
MatMul roadmap.  QNN Native is the specification source; the future owned
runtime must not execute through QNN, but it must match these raw outputs,
runtime contracts, and performance scopes.

Machine-readable manifest: `../../example/handwritten_hmx_matmul/oracles.json`.
W16A16 remains in the manifest as retained evidence, but it is inactive for the
current W4A16 direct-body custom-baseline route.

Regenerate and validate:

```bash
uv run python scripts/build_handwritten_oracle_manifest.py
uv run python scripts/check_handwritten_oracle_manifest.py
```

## Canonical Oracles

| Family | Native oracle | Custom comparator | Shape/chain | Kernel event | QNN aggregate | Timeline | Exactness |
|---|---|---|---|---:|---:|---:|---|
| u8i8 | `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256` | `example/qnn_matmul_profile/output_u8i8_aligned_e2e_256` | `[256, 256, 256]`, chain `8` | 12435 | 36922 | 53946 | 65536/65536, maxdiff 0 |
| w4a8 | `example/qnn_matmul_profile/output_w4a8_native_ref_e2e_256` | `example/qnn_matmul_profile/output_w4a8_aligned_e2e_256` | `[256, 256, 256]`, chain `8` | 11546 | 29765 | 48831 | 65536/65536, maxdiff 0 |
| w8a16 | `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256` | `example/qnn_matmul_profile/output_w8a16_aligned_e2e_256` | `[256, 256, 256]`, chain `8` | 30182 | 35747 | 79095 | 65536/65536, maxdiff 0 |
| w4a16 | `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256` | `example/qnn_matmul_profile/output_w4a16_aligned_e2e_256` | `[256, 256, 256]`, chain `8` | 29815 | 70408 | 253245 | 65536/65536, maxdiff 0 |
| w16a16 | `example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256` | `example/qnn_matmul_profile/output_w16a16_accepted_256` | `[256, 256, 256]`, chain `1` | 75433 | 82644 | 124593 | native_raw_exact=True |

## Acceptance Evidence

- All five native oracle directories are the retained canonical directories under
  `example/qnn_matmul_profile/`.
- Each native oracle passed:

```bash
scripts/check_qnn_artifact_standard.py <native_dir> \
  --require-native-io --require-layout-flags --reject-float-io
```

- The manifest records raw input/output files, SHA256, storage type, byte
  count, shape, chain, quantization mode, prepared payload sources, native
  HMX body name, native compute contracts for activation, packed weight,
  folded bias, control, extra/control, and output tensors, comparable
  native optrace events, QNN-op aggregate cycles, packet counts, timeline
  span, public raw-output exactness scope, and exactness evidence.
- `scripts/check_handwritten_oracle_manifest.py` validates the generated JSON
  against this Milestone-0 contract.
- Stale probe directories and same-shape random native runs are excluded.

## Family Notes

### u8i8

- Op surface: `HmxU8I8ToU8MatMul`.
- Native HMX body: `hmx_v73_convbbb1x1deep_stride1`.
- Raw input: `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256/runtime_inputs_native/A.raw` (65536 bytes).
- Raw output oracle: `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256/device_out/Y.raw` (65536 bytes).
- Native kernel event: `q::ConvLayer_s1.opt`; 8 events, 12435 cycles, packets `[346, 346, 346, 346, 346, 346, 346, 346]`.
- QNN aggregate prefix `MatMul_` sums to 36922 cycles; timeline span 53946 cycles.
- Comparison scope: public output exactness target is `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256/device_out/Y.raw` (65536 bytes); native kernel scope has 8 `q::ConvLayer_s1.opt` events and 524288 total output-scope bytes.
- Native kernel node count check: observed 8, expected 8, pass True.

### w4a8

- Op surface: `HmxU8I4ToU8MatMul`.
- Native HMX body: `hmx_v73_convbnb1x1_stride1`.
- Raw input: `example/qnn_matmul_profile/output_w4a8_native_ref_e2e_256/runtime_inputs_native/A.raw` (65536 bytes).
- Raw output oracle: `example/qnn_matmul_profile/output_w4a8_native_ref_e2e_256/device_out/Y.raw` (65536 bytes).
- Native kernel event: `q::ConvLayer_s1.opt`; 8 events, 11546 cycles, packets `[330, 330, 330, 330, 330, 330, 330, 330]`.
- QNN aggregate prefix `MatMul_` sums to 29765 cycles; timeline span 48831 cycles.
- Comparison scope: public output exactness target is `example/qnn_matmul_profile/output_w4a8_native_ref_e2e_256/device_out/Y.raw` (65536 bytes); native kernel scope has 8 `q::ConvLayer_s1.opt` events and 524288 total output-scope bytes.
- Native kernel node count check: observed 8, expected 8, pass True.

### w8a16

- Op surface: `HmxU16I8ToU16MatMul`.
- Native HMX body: `hmx_v75_convhbh1x1deep_stride1`.
- Raw input: `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/runtime_inputs_native/A.raw` (131072 bytes).
- Raw output oracle: `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes).
- Native kernel event: `q::ConvLayer_s1.opt`; 8 events, 30182 cycles, packets `[695, 695, 695, 695, 695, 695, 695, 695]`.
- QNN aggregate prefix `matmul_` sums to 35747 cycles; timeline span 79095 cycles.
- Comparison scope: public output exactness target is `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes); native kernel scope has 8 `q::ConvLayer_s1.opt` events and 1048576 total output-scope bytes.
- Native kernel node count check: observed 8, expected 8, pass True.

### w4a16

- Op surface: `HmxU16I4ToU16MatMul`.
- Native HMX body: `hmx_v73_convhnh1x1_stride1`.
- Raw input: `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/runtime_inputs_native/A.raw` (131072 bytes).
- Raw output oracle: `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes).
- Native kernel event: `q::ConvLayer_s1.opt`; 8 events, 29815 cycles, packets `[808, 808, 808, 808, 808, 808, 808, 808]`.
- QNN aggregate prefix `conv1x1_` sums to 70408 cycles; timeline span 253245 cycles.
- Comparison scope: public output exactness target is `example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes); native kernel scope has 8 `q::ConvLayer_s1.opt` events and 1048576 total output-scope bytes.
- Native kernel node count check: observed 8, expected 8, pass True.

### w16a16

- Op surface: `HmxU16I16ToU16MatMul`.
- Native HMX body: `hmx_v73_convhhh1x1_stride1`.
- Raw input: `example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/runtime_inputs_native/A.raw` (131072 bytes).
- Raw output oracle: `example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes).
- Native kernel event: `q::ConvLayer_s1.opt`; 2 events, 75433 cycles, packets `[8836, 8836]`.
- QNN aggregate prefix `matmul_` sums to 82644 cycles; timeline span 124593 cycles.
- Comparison scope: public output exactness target is `example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/device_out/Y.raw` (131072 bytes); native kernel scope has 2 `q::ConvLayer_s1.opt` events and 131072 total output-scope bytes.
- Native kernel node count check: observed 2, expected 2, pass True.
