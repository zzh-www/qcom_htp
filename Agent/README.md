# Agent Knowledge

This directory is the only home for agent-managed project knowledge in
`qcom_htp`.  Keep this README as an entrypoint: add links here, put details in
the linked documents.

## Start Here

| Document | Purpose |
|---|---|
| [w8a16 native alignment handoff](handoffs/w8a16_native_alignment.md) | Completed 256^3 `HmxU16I8ToU16MatMul` native-output and performance alignment; includes reproduction command and next-round guidance. |
| [w4a16 QNN native path](handoffs/w4a16_qnn_native_path.md) | Required native-first entrypoint for W4A16: Conv lowering path, tensor contracts, sidecars, optrace scopes, and skel wrapper flow. |
| [w4a16 native alignment handoff](handoffs/w4a16_native_alignment.md) | Current `HmxU16I4ToU16MatMul` native-alignment flow, standard optrace artifacts, probe results, and remaining blocker. |
| [QNN native artifact standard](current/qnn_native_artifact_standard.md) | Required DLC, context-binary, native I/O, and optrace/performance artifact flow for QNN-native references and custom-op comparisons. |
| [QNN HMX MatMul status](current/qnn_hmx_matmul_status.md) | Broader quantized MatMul family status, design notes, performance-reading rules, and historical probe evidence. |
| [QNN reverse-engineering evidence](qnn_re/) | Non-Markdown disassembly slices and large cached evidence such as `skel_text_full.S`. |

## Rules

- Keep `Agent/README.md` short and link-oriented.
- Put handoffs under `Agent/handoffs/`.
- Put current status and longer working notes under `Agent/current/`.
- Keep reverse-engineering evidence under `Agent/qnn_re/`.
- Do not add agent-managed docs under repo root, `docs/`, or ad-hoc folders.
