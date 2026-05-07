# qnn_hmx_matmul_w4a16

Independent QNN custom-op package for `HmxU16I4ToU16MatMul`, the `w4a16`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u16` activation/output and a direct byte-carrier
  weight payload; logical signed W4 packing and LPBQ/per-group extensions are
  owned by this family.
- Device builds currently define `HMX_W4A16_SKIP_KERNEL`, so the callback writes
  the marker path until the real A16/W4 HMX body and packed-weight ABI are
  validated.

Build:

```bash
bash example/qnn_hmx_matmul_w4a16/build.sh
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w4a16_chain.sh
```
