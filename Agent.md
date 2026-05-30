# Agent.md

`Agent/` is the repo-local home for agent-managed project knowledge.

Start here: [Agent/README.md](Agent/README.md).

Hard rules:

- Use `uv` for Python environment management and Python script execution.
- Before every formal code commit or push, run the full kernel E2E gate.  For
  pushes, use `scripts/run_kernel_ci_preflight.sh` or
  `scripts/push_with_kernel_ci.sh`; these run `tests/qnn_kernel_e2e/run_all.sh`
  before Git opens the remote push connection.
- Enable the repo-local pre-push hook with `scripts/install_git_hooks.sh` in
  each checkout.  The hook only accepts pushed refs whose commit and tree match
  the latest recorded kernel CI preflight proof.
- Keep new agent-managed docs under `Agent/`; do not add them under repo root,
  `docs/`, or ad-hoc folders.
- Keep non-Markdown reverse-engineering evidence under `Agent/qnn_re/`.

Current implementation boundary:

- QNN custom/native kernel correctness is tracked by
  [Agent/current/qnn_hmx_matmul_status.md](Agent/current/qnn_hmx_matmul_status.md)
  and [Agent/current/qnn_kernel_e2e_ci.md](Agent/current/qnn_kernel_e2e_ci.md).
- The handwritten HMX MatMul route is QNN-free at runtime and lives under
  `example/handwritten_hmx_matmul/`.
- Do not restart the old W4A16 QNN blackbox/tutorial-wrapper route as the
  handwritten implementation plan; it is retained only as historical
  provenance in `Agent/handoffs/`.
