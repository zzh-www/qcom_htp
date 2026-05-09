# qnn_hmx_matmul_w16a16

Independent QNN custom-op package for `HmxU16I16ToU16MatMul`, the `w16a16`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u16` activation/output and a direct `u16` carrier
  for logical signed W16 weights; per-channel scale/bias data belongs in the
  native prepared payload.
- Device builds currently define `HMX_W16A16_SKIP_KERNEL`, so the callback writes
  the marker path until the real A16/W16 HMX body and packing contract are
  validated.
- `W16A16_KERNEL_PROFILE=accepted` is the current opt-in real-kernel profile for
  the canonical native 256^3 oracle.  It auto-generates byte-equivalent prepared
  weight and bias sidecars from `matmul.onnx`, reproduces native raw output
  exactly, and is native-class on kernel cycles/packets.  Default builds remain
  skip-guarded because the accepted native-record fields are scoped to 256^3 and
  the public custom boundary is one tiled op with an internal N128 split rather
  than native's two Conv graph nodes.
- The standard conversion path is encoding-driven
  `qairt-converter -> qairt-quantizer`: converter applies generated encodings,
  then quantizer runs without calibration input or a custom op package so CPU
  backend never executes the custom op during quantization.

Build:

```bash
bash example/qnn_hmx_matmul_w16a16/build.sh
bash example/qnn_hmx_matmul_w16a16/build_x86.sh
```

Accepted native-record profile:

```bash
W16A16_KERNEL_PROFILE=accepted \
  bash example/qnn_hmx_matmul_w16a16/build.sh
W16A16_KERNEL_PROFILE=accepted \
  bash example/qnn_hmx_matmul_w16a16/build_x86.sh

W16A16_KERNEL_PROFILE=accepted \
W16A16_NATIVE_ORACLE_DIR="$PWD/example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256" \
OUT_DIR="$PWD/example/qnn_matmul_profile/output_w16a16_accepted_256" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq NATIVE_OUTPUT=1 \
uv run bash example/qnn_hmx_matmul_w16a16/standard_flow/custom_w16a16/run_w16a16_chain.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w16a16/standard_flow/custom_w16a16
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w16a16_chain.sh
```
