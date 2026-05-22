# Agent Knowledge

This directory is the only home for agent-managed project knowledge in
`qcom_htp`.  Keep this README as an entrypoint: add links here, put details in
the linked documents.

## Start Here

| Document | Purpose |
|---|---|
| [Handwritten HMX MatMul roadmap](guides/handwritten_hmx_matmul_roadmap.md) | Current QNN-free runtime route, accepted active families, W4A16 chain8 custom-baseline bridge, and completion gates. |
| [Handwritten HMX MatMul runtime boundary](current/handwritten_hmx_matmul_runtime.md) | Current owned runtime files, device gate, and W4A16 custom-baseline/native-bridge boundary. |
| [Handwritten HMX MatMul body evidence](current/handwritten_hmx_matmul_bodies.md) | Body manifest and byte-identity evidence for owned HMX kernel slices. |
| [Handwritten HMX MatMul oracles](current/handwritten_hmx_matmul_oracles.md) | QNN Native oracle freeze used only for prepared bytes, raw outputs, and perf references. |
| [QNN HMX MatMul status](current/qnn_hmx_matmul_status.md) | Concise current direction and historical boundary for quantized MatMul family work. |
| [w16a16 native alignment plan](handoffs/w16a16_native_alignment_plan.md) | Plan for aligning `HmxU16I16ToU16MatMul`: native oracle, split-N kernel surface, body verification, probes, and acceptance gates. |
| [w8a16 native alignment handoff](handoffs/w8a16_native_alignment.md) | Current 256^3 `HmxU16I8ToU16MatMul` state: native-output exact, but kernel-entry shape and performance remain open. |
| [w4a16 QNN native path](handoffs/w4a16_qnn_native_path.md) | Historical/native-oracle entrypoint for W4A16: Conv lowering path, tensor contracts, sidecars, optrace scopes, and skel wrapper flow. It is not the active implementation route. |
| [w4a16 native alignment handoff](handoffs/w4a16_native_alignment.md) | Current `HmxU16I4ToU16MatMul` probe history: QNN custom chain8 remains transpose-aware exact, while active W4A16 work has moved to the direct-body custom-baseline route under `example/handwritten_hmx_matmul/`. |
| [w4a16 Python formula mismatch report](../docs/w4a16_python_formula_mismatch_report.html) | HTML report explaining the W4A16 HMX floor256 accumulator drain required for Python/native bit-exactness. |
| [QNN native artifact standard](current/qnn_native_artifact_standard.md) | Required DLC, context-binary, native I/O, and optrace/performance artifact flow for QNN-native references and custom-op comparisons. |
| [QNN kernel E2E CI](current/qnn_kernel_e2e_ci.md) | Required kernel CI entrypoints, including handwritten HMX MatMul, and the formal pre-push device gate. |
| [QNN HTP per-channel bias prepare](guides/qnn_htp_perchannel_bias_prepare.md) | Recovered HTP prepare algorithm for DLC Conv `B` to final per-channel `bias_to_vtcm` sidecar bias, plus public Python API and validation commands. |
| [QNN HTP W8A16 per-channel sidecar](guides/qnn_htp_w8a16_perchannel_sidecar.md) | W8A16 generated sidecar ABI for custom HMX, how it differs from QNN Native final sidecar, and the promoted correctness CI gate. |
| [QNN native alignment blackbox handbook](guides/qnn_native_alignment_blackbox_handbook.md) | Methodology for future QNN-native blackbox alignment. For W4A16 handwritten MatMul this is superseded by the direct-body custom-baseline route. |
| [QNN reverse-engineering evidence](qnn_re/) | Non-Markdown disassembly slices and large cached evidence such as `skel_text_full.S`. |

## Rules

- Keep `Agent/README.md` short and link-oriented.
- Put durable agent methodology under `Agent/guides/`.
- Put handoffs under `Agent/handoffs/`.
- Put current status and longer working notes under `Agent/current/`.
- Keep reverse-engineering evidence under `Agent/qnn_re/`.
- Do not add agent-managed docs under repo root, `docs/`, or ad-hoc folders.
