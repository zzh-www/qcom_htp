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
- Default device builds enter the real W4A8 HMX body.  The canonical 256^3
  chain8 output is byte-exact against the matched native oracle
  (`65536/65536`, maxdiff `0`).
- Performance is aligned for the canonical 256^3 chain8 artifact.  Optrace
  reports custom main `10025` cycles and timeline `38644`; matched native
  reports `q::ConvLayer_s1.opt=11546`, MatMul aggregate `29765`, and timeline
  `48831`.
- LPBQ is supported as an explicit build/run profile.  `LPBQ_ONLY=1` registers
  the 9-input Crouton HMX signature used by QAIRT v1 LPBQ side tensors, while
  `LPBQ_SCALAR=1` registers a direct Flat4 scalar correctness fallback.
  Canonical 256^3 chain8 LPBQ fast validation is byte-exact against the matched
  native oracle (`65536/65536`) with custom main `9629` cycles.

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

LPBQ fast validation:

```bash
LPBQ_ONLY=1 bash example/qnn_hmx_matmul_w4a8/build.sh
LPBQ_ONLY=1 bash example/qnn_hmx_matmul_w4a8/build_x86.sh

W4_ENCODING=lpbq CHAIN=8 M=256 K=256 N=256 \
OUT_DIR=/tmp/qcom_htp_w4a8_lpbq_hmx_chain8 \
bash example/qnn_hmx_matmul_w4a8/standard_flow/custom_w4a8/run_w4a8_chain.sh
```
