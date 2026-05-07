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

Build:

```bash
bash example/qnn_hmx_matmul_w16a16/build.sh
bash example/qnn_hmx_matmul_w16a16/build_x86.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w16a16/standard_flow/custom_w16a16
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w16a16_chain.sh
```
