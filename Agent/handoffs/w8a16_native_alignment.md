# w8a16 Native Alignment Handoff

Current status: `example/qnn_hmx_matmul_w8a16`
(`HmxU16I8ToU16MatMul`, i8 weight x u16 activation -> u16 output) is aligned
for the canonical 256^3 chain8 graph.  Custom and native both enter the HTP
kernel with activation/output `UFixed16 [1,8,32,256]`, the custom output is
byte-identical to the matched QNN native oracle, and kernel/timeline
performance is native-class.

Final acceptance rule:

1. Correctness is judged against real QNN native output/artifacts for the same
   graph/input. Analytic formulas are diagnostics only.
2. Performance evidence uses the standard optrace artifact set under
   `<out_dir>/optrace/`; do not cite ad-hoc `/tmp/_optrace*` files as the
   durable source.
3. Report both HMX kernel-only and HTP graph timeline-span numbers. Kernel-only
   is useful for body/descriptor parity; timeline span catches sidecar/layout
   regressions.

Current verified result:

| Check | Custom | Native | Status |
|---|---:|---:|---|
| output vs native artifact | `65536/65536`, maxdiff `0` | oracle | aligned |
| kernel-entry activation/output shape | `UFixed16 [1,8,32,256]` on 8 nodes | `UFixed16 [1,8,32,256]` on 8 nodes | aligned |
| HMX kernel cycles | `30871` | `30182` | aligned |
| HTP graph timeline span | `80217` | `79095` | aligned |

Durable artifacts:

| Artifact | Path |
|---|---|
| custom run | `example/qnn_matmul_profile/output_w8a16_aligned_e2e_256/` |
| native oracle | `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw` |
| custom optrace | `example/qnn_matmul_profile/output_w8a16_aligned_e2e_256/optrace/` |
| native optrace | `example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/optrace/` |

The solution path that mattered:

1. Fix the oracle first. `qnn-net-run` may emit dequantized float output; the
   verifier now re-quantizes using `quant_overrides.json`. When
   `VERIFY_NATIVE_RAW` is set, native byte equality is the primary pass/fail
   check and analytic output is only printed as a diagnostic.
2. Current final artifact uses the executable native-tiled graph:
   `MODE=chain_qdq --op-input-layout tiled --final-output-rank 3d
   --a16-quant-contract native --w8-pack-order kmajor --bias-layout native_a16
   --reference-contract native`.
3. Keep the validated native data contracts: full-W8 K-major 32x32 tile order,
   native A16 output encoding, native-shaped 512B bias records, and the
   three-word control table `[1, 1025, 524]`.
4. The final performance fix was descriptor/table-contract alignment, not
   another data packing tweak.  Native-class cycles require physical Crouton
   table entries and `n_tiles_pow2=M_t*4`.  The obsolete row-expanded probes
   were deleted after proving only that over-expanded tables can be correct but
   non-native in packet count.
5. Treat old native-rank, wrapper-bundle, descriptor-dump, and table-window
   probe artifacts as superseded.  Their retained conclusion is only that the
   aligned tiled physical-table contract is the native-class default.
6. Default build scripts now compile the real HMX body.  `HMX_W8A16_SKIP_KERNEL`
   remains available only as an explicit diagnostic override.

Reproduction command:

```bash
bash example/qnn_hmx_matmul_w8a16/build.sh
bash example/qnn_hmx_matmul_w8a16/build_x86.sh

VERIFY_NATIVE_RAW="$PWD/example/qnn_matmul_profile/output_w8a16_native_ref_e2e_256/device_out/Y.raw" \
OUT_DIR="$PWD/example/qnn_matmul_profile/output_w8a16_aligned_e2e_256" \
M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq \
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
  high-`m_total` production candidates.  The old local probe artifacts were
  removed; regenerate them only with a new hypothesis and a fresh artifact name.
- Current follow-up: extend the aligned tiled contract beyond the canonical
  256^3 chain8 case.  Split-N public QHPI output and broader shape coverage are
  still separate graph-contract issues.
- Native split-N public QHPI output remains a separate graph-contract issue and
  is outside the cleaned canonical artifact set.
