# qnn_hmx_matmul_w4a16

Independent QNN custom-op package for `HmxU16I4ToU16MatMul`, the `w4a16`
MatMul/FullyConnected-as-Conv1d family.

Current status:

- Package/provider/op XML, converter hook, build scripts, and custom graph flow
  are separate from `u8i8`.
- QHPI ABI uses per-tensor `u16` activation/output and a direct byte-carrier
  weight payload; logical signed W4 packing and LPBQ/per-group extensions are
  owned by this family.
- The canonical 256^3 native-aligned path is bit-exact against QNN native with
  `HMX_W4A16_NATIVE_COMPACT_SOURCE_TABLES`, native A16 bias/control, and
  `native_kblock32_nmajor_k4_lohi` W4 packing.  Default builds still define
  `HMX_W4A16_SKIP_KERNEL`; remove it explicitly for real-kernel validation.
- The QNN-native Conv reference may keep a float ONNX public surface, but the
  accepted comparison artifact is the u16 runtime contract in `native_io.json`
  plus `--use_native_input_files --use_native_output_files` and NONTRIVIAL
  converter layout flags.
- Native-alignment process and optrace artifacts are tracked in
  [Agent/handoffs/w4a16_native_alignment.md](../../Agent/handoffs/w4a16_native_alignment.md).

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
