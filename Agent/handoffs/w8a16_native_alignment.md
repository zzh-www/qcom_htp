# w8a16 Native Alignment Handoff

Completed milestone: `example/qnn_hmx_matmul_w8a16`
(`HmxU16I8ToU16MatMul`, i8 weight x u16 activation -> u16 output) is aligned
with QNN native for the canonical 256^3 native-rank path.

Final acceptance rule:

1. Correctness is judged against real QNN native output/artifacts for the same
   graph/input. Analytic formulas are diagnostics only.
2. Performance evidence uses the standard optrace artifact set under
   `<out_dir>/optrace/`; do not cite ad-hoc `/tmp/_optrace*` files as the
   durable source.
3. Report both HMX kernel-only and HTP graph timeline-span numbers. Kernel-only
   is useful for body/descriptor parity; timeline span catches sidecar/layout
   regressions.

Final verified result:

| Check | Custom | Native | Status |
|---|---:|---:|---|
| output vs native artifact | `65536/65536`, maxdiff `0` | oracle | aligned |
| HMX kernel cycles | `29842` | `17675 + 13164 = 30839` | aligned |
| HMX packets | `4938` | about `5086` | aligned |
| HTP graph timeline span | `80002` | `80887` | aligned |

Durable artifacts:

| Artifact | Path |
|---|---|
| custom run | `example/qnn_matmul_profile/output_codex_w8a16_native_rank_fast_default_chainqdq_final3d_precompute_256/` |
| native oracle | `example/qnn_matmul_profile/output_codex_native_w8a16_custom_full_256/device_out/out.raw` |
| custom optrace | `example/qnn_matmul_profile/output_codex_w8a16_native_rank_fast_default_chainqdq_final3d_precompute_256/optrace/` |
| native optrace | `example/qnn_matmul_profile/output_codex_native_w8a16_custom_full_256/optrace/` |

The solution path that mattered:

1. Fix the oracle first. `qnn-net-run` may emit dequantized float output; the
   verifier now re-quantizes using `quant_overrides.json`. When
   `VERIFY_NATIVE_RAW` is set, native byte equality is the primary pass/fail
   check and analytic output is only printed as a diagnostic.
2. Use the executable native-rank graph:
   `MODE=chain_qdq --op-input-layout native --final-output-rank 3d
   --a16-quant-contract native --w8-pack-order kmajor --bias-layout native_a16
   --reference-contract native`.
3. Keep the validated native data contracts: full-W8 K-major 32x32 tile order,
   native A16 output encoding, native-shaped 512B bias records, and the
   three-word control table `[1, 1025, 524]`.
4. The actual semantic fix was descriptor/mask alignment, not another data
   packing tweak:
   `HMX_W8A16_MASK_ARG1=0x70b`, `n_tiles_pow2=row4_groups*4`,
   `m_total_minus_step=8`.
5. The temporary `m_total_minus_step=456` workaround proved the row coverage
   hypothesis but was slow (`185183` cycles, about `35178` packets). Replacing
   it with `n_tiles_pow2=row4_groups*4` restored native-class performance.
6. Preserve guarded builds. Default build scripts still define
   `HMX_W8A16_SKIP_KERNEL`; real-kernel validation must explicitly pass
   `EXTRA_DEFS="-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL"`.

Reproduction command:

```bash
EXTRA_DEFS="-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w8a16/build.sh
EXTRA_DEFS="-UHMX_W8A16_SKIP_KERNEL -DHMX_W8A16_ALLOW_UNVALIDATED_KERNEL" \
bash example/qnn_hmx_matmul_w8a16/build_x86.sh

VERIFY_NATIVE_RAW="$PWD/example/qnn_matmul_profile/output_codex_native_w8a16_custom_full_256/device_out/out.raw" \
OUT_DIR="$PWD/example/qnn_matmul_profile/output_codex_w8a16_native_rank_fast_default_chainqdq_final3d_precompute_256" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq \
GEN_EXTRA_ARGS="--op-input-layout native --final-output-rank 3d --a16-quant-contract native --w8-pack-order kmajor --bias-layout native_a16 --reference-contract native" \
bash example/qnn_hmx_matmul_w8a16/standard_flow/custom_w8a16/run_w8a16_chain.sh
```

Next-round guidance:

- Start new OP work by making the native oracle artifact first, then require
  custom output equality to that artifact. Do not promote analytic references
  to acceptance criteria.
- Generate standard performance artifacts with
  `scripts/decode_qnn_optrace.py <out_dir>` or let the chain runner do it.
  Cite `optrace/summary.json` and `optrace/chrometrace.json`.
- Do not repeat the w8a16 dead ends unless new evidence appears: signed W8
  carrier flipping, bias-only sweeps, low-pattern native artifacts, direct raw
  output, `Layout_Any` / `Layout_Custom`, old `0x700` mask sweeps, or
  high-`m_total` production candidates.
- Native split-N public QHPI output remains a separate graph-contract issue:
  one `[1,1,256,256]` custom output plus final-3D export works; two
  `[1,1,256,128]` custom outputs still fail before useful callback semantics.
