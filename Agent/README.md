# Agent Knowledge

This directory is the only home for agent-managed project knowledge in
`qcom_htp`.  Keep this README as an entrypoint: add links here, put details in
the linked documents.

## Start Here

| Document | Purpose |
|---|---|
| [w16a16 native alignment plan](handoffs/w16a16_native_alignment_plan.md) | Plan for aligning `HmxU16I16ToU16MatMul`: native oracle, split-N kernel surface, body verification, probes, and acceptance gates. |
| [w8a16 native alignment handoff](handoffs/w8a16_native_alignment.md) | Current 256^3 `HmxU16I8ToU16MatMul` state: native-output exact, but kernel-entry shape and performance remain open. |
| [w4a16 QNN native path](handoffs/w4a16_qnn_native_path.md) | Required native-first entrypoint for W4A16: Conv lowering path, tensor contracts, sidecars, optrace scopes, and skel wrapper flow. |
| [w4a16 native alignment handoff](handoffs/w4a16_native_alignment.md) | Completed 256^3 `HmxU16I4ToU16MatMul` native-output and performance alignment, plus standard optrace artifacts and probe history. |
| [w4a16 Python formula mismatch report](../docs/w4a16_python_formula_mismatch_report.html) | HTML report explaining the W4A16 HMX floor256 accumulator drain required for Python/native bit-exactness. |
| [QNN native artifact standard](current/qnn_native_artifact_standard.md) | Required DLC, context-binary, native I/O, and optrace/performance artifact flow for QNN-native references and custom-op comparisons. |
| [QNN kernel E2E CI](current/qnn_kernel_e2e_ci.md) | Required per-kernel E2E CI entrypoints and the formal pre-push device gate. |
| [QNN native alignment blackbox handbook](guides/qnn_native_alignment_blackbox_handbook.md) | Methodology for future QNN-native blackbox alignment: oracle setup, probes, negative evidence, payload checks, and completion gates. |
| [QNN HMX MatMul status](current/qnn_hmx_matmul_status.md) | Broader quantized MatMul family status, design notes, performance-reading rules, and historical probe evidence. |
| [QNN reverse-engineering evidence](qnn_re/) | Non-Markdown disassembly slices and large cached evidence such as `skel_text_full.S`. |

## Rules

- Keep `Agent/README.md` short and link-oriented.
- Put durable agent methodology under `Agent/guides/`.
- Put handoffs under `Agent/handoffs/`.
- Put current status and longer working notes under `Agent/current/`.
- Keep reverse-engineering evidence under `Agent/qnn_re/`.
- Do not add agent-managed docs under repo root, `docs/`, or ad-hoc folders.
