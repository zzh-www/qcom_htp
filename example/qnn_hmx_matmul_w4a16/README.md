# qnn_hmx_matmul_w4a16

Independent QNN custom-op package for `HmxU16I4ToU16MatMul`, the `w4a16`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u16` activation/output and a direct byte-carrier
  weight payload; logical signed W4 packing and LPBQ/per-group extensions are
  owned by this family.
- The canonical 256^3 chain8 native-aligned path is bit-exact against QNN
  native with `HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES`, native A16
  bias/control, and `native_kblock32_nmajor_k4_lohi` W4 packing.  Custom and
  native both enter each HTP kernel with activation shape `(1,8,32,256)`.
  Default builds still define `HMX_W4A16_SKIP_KERNEL`; remove it explicitly for
  real-kernel validation.
- The QNN-native Conv reference may keep a float ONNX public surface, but the
  accepted comparison artifact is the u16 runtime contract in `native_io.json`
  plus `--use_native_input_files --use_native_output_files` and NONTRIVIAL
  converter layout flags.
- W4 conversion defaults to an encoding-driven
  `qairt-converter -> qairt-quantizer` path with `--pack_4_bit_weights`.
  Converter applies generated encodings and quantizer runs without calibration
  input or a custom op package; set `PACK_4BIT_WEIGHTS=0` only for explicit
  legacy carrier probes.
- Native-alignment process and optrace artifacts are tracked in
  [Agent/handoffs/w4a16_native_alignment.md](../../Agent/handoffs/w4a16_native_alignment.md).
- LPBQ is supported as an explicit build/run profile.  `LPBQ_ONLY=1` registers
  the 9-input Crouton HMX signature and lets the LPBQ path bypass the default
  skip guard; `LPBQ_SCALAR=1` registers a direct Flat4 scalar correctness
  fallback.  Canonical 256^3 chain8 LPBQ fast validation is native-exact
  (`65536/65536` after the same native-output transpose used by the standard
  W4A16 gate).  Current live repeats show native-class performance: custom
  LPBQ main `30820`/`30991` cycles after one cold-side `33744` run, versus
  native W4A16 Conv repeats `32563`/`33196` cycles.  The retained native
  artifact still records a low-side `29815` sample and should be treated as a
  historical single-run measurement, not as the only performance oracle.

Build:

```bash
bash example/qnn_hmx_matmul_w4a16/build.sh
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
```

Native-aligned real-kernel build:

```bash
EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build.sh

EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w4a16/build_x86.sh
```

Smoke flow:

```bash
cd example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16
SKIP_DEVICE=1 M=32 K=32 N=32 CHAIN=1 bash run_w4a16_chain.sh
```

Canonical 256^3 validation uses the final artifact directories under
`example/qnn_matmul_profile/`: regenerate
`output_w4a16_native_ref_e2e_256/` with `CHAIN=8
run_native_w4a16_conv_ref.sh`, then run this custom flow with `CHAIN=8` and
`VERIFY_NATIVE_RAW` pointing at the native `device_out/Y.raw`.

LPBQ fast validation:

```bash
LPBQ_ONLY=1 bash example/qnn_hmx_matmul_w4a16/build.sh
LPBQ_ONLY=1 bash example/qnn_hmx_matmul_w4a16/build_x86.sh

W4_ENCODING=lpbq CHAIN=8 MODE=chain_qdq M=256 K=256 N=256 \
VERIFY_NATIVE_RAW=/home/zzh/work/qcom_htp/example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256/device_out/Y.raw \
VERIFY_NATIVE_TRANSPOSE=1 \
OUT_DIR=/tmp/qcom_htp_w4a16_lpbq_hmx_chain8 \
bash example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh
```
