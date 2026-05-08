# qnn_hmx_matmul_w4a8

Independent QNN custom-op package for `HmxU8I4ToU8MatMul`, the `w4a8`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u8` activation/output and a direct byte-carrier
  weight payload; logical signed W4 packing is owned by the family packer.
- W4 conversion defaults to an encoding-driven
  `qairt-converter -> qairt-quantizer` path with `--pack_4_bit_weights`.
  Converter applies generated encodings and quantizer runs without calibration
  input or a custom op package; set `PACK_4BIT_WEIGHTS=0` only for explicit
  legacy carrier probes.
- Device builds currently define `HMX_W4A8_SKIP_KERNEL`, so the callback writes
  the marker path until the real W4 HMX body and packed-weight ABI are validated.

Build:

```bash
bash example/qnn_hmx_matmul_w4a8/build.sh
bash example/qnn_hmx_matmul_w4a8/build_x86.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w4a8/standard_flow/custom_w4a8
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w4a8_chain.sh
```
