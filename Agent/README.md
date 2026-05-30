# Agent Knowledge

This directory is the home for agent-managed project knowledge in `qcom_htp`.
Use this README as a map only; keep detailed evidence in the linked files.

## Positioning

`Agent/` is the formal, systematic, **in-repo** knowledge base for this project.
Because it lives in git, it is **shared across every agent and checkout** — any
agent working in `qcom_htp` reads and contributes to the same `Agent/` tree.  It
is the authoritative source of project state: when anything disagrees,
`Agent/current/` wins.

This is distinct from an agent's *private auto-memory* (e.g. files under a
local `~/.claude/.../memory/` dir).  Private memory is per-agent, not shipped,
and only a terse recall layer — it should *point into* `Agent/` rather than
duplicate, override, or substitute for it.  Durable, substantive project
knowledge belongs here in `Agent/`, written so any other agent can pick it up.

## Current Fact Sources

| Document | Use |
|---|---|
| [QNN HMX MatMul status](current/qnn_hmx_matmul_status.md) | Current kernel-family state, promoted gates, LPBQ status, and historical boundaries. |
| [QNN kernel E2E CI](current/qnn_kernel_e2e_ci.md) | Formal CI entrypoints, promoted leaf gates, pre-push behavior, and result matrices. |
| [QNN native artifact standard](current/qnn_native_artifact_standard.md) | Required DLC/context/native-I/O/optrace artifact layout for QNN Native and custom-op comparisons. |
| [Handwritten HMX MatMul runtime boundary](current/handwritten_hmx_matmul_runtime.md) | Current QNN-free runtime tree, accepted families, and formal gate. |
| [Handwritten HMX MatMul body evidence](current/handwritten_hmx_matmul_bodies.md) | Owned HMX body slices and byte-identity evidence. |
| [Handwritten HMX MatMul oracles](current/handwritten_hmx_matmul_oracles.md) | Frozen QNN Native oracle artifacts used by the handwritten runtime. |

## Durable Guides

| Document | Use |
|---|---|
| [Handwritten HMX MatMul roadmap](guides/handwritten_hmx_matmul_roadmap.md) | QNN-free handwritten route, completion criteria, and W4A16 custom-baseline bridge. |
| [QNN HTP per-channel bias prepare](guides/qnn_htp_perchannel_bias_prepare.md) | Recovered HTP prepare algorithm for per-channel Conv bias sidecars. |
| [QNN HTP W8A16 per-channel sidecar](guides/qnn_htp_w8a16_perchannel_sidecar.md) | W8A16 custom sidecar ABI and generated-sidecar implementation notes. |
| [QNN native alignment blackbox handbook](guides/qnn_native_alignment_blackbox_handbook.md) | General blackbox methodology for future QNN-native alignment work. |

## Historical Handoffs

These files preserve investigation history.  Do not treat them as the current
implementation route unless a current fact source links to a specific section.

| Document | Scope |
|---|---|
| [w16a16 native alignment plan](handoffs/w16a16_native_alignment_plan.md) | Open W16A16 native-alignment plan; W16A16 is not in the active gate. |
| [w8a16 native alignment handoff](handoffs/w8a16_native_alignment.md) | W8A16 native-output exactness history and remaining kernel-entry/perf questions. |
| [w4a16 QNN native path](handoffs/w4a16_qnn_native_path.md) | Historical W4A16 QNN Native/custom-op route and provenance. |
| [w4a16 native alignment handoff](handoffs/w4a16_native_alignment.md) | Historical W4A16 probe log; active W4A16 acceptance moved to the handwritten custom-baseline route. |
| [w8a16 zp-neutral optrace](handoffs/w8a16_zp_neutral_optrace.md) | Narrow archived blocker note for W8A16 sidecar/debug provenance. |

## Evidence

`Agent/qnn_re/` contains non-Markdown reverse-engineering evidence and
disassembly slices.  Keep large raw evidence there rather than in status docs.

## Rules

- Keep `Agent/README.md` short and link-oriented.
- Put stable methods under `Agent/guides/`.
- Put current state under `Agent/current/`.
- Put historical handoff/probe logs under `Agent/handoffs/`.
- Keep reverse-engineering evidence under `Agent/qnn_re/`.
- Do not add new agent-managed documents under repo root, `docs/`, or ad-hoc folders.
