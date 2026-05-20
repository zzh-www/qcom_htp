# Agent.md

`Agent/` is the only home for agent-managed project knowledge in this repo.

Start here: [Agent/README.md](Agent/README.md).

Rules:
- Keep this file short.
- Keep `Agent/README.md` as a short link-oriented entrypoint.
- Put current status, handoffs, findings, and project-specific rules in linked documents under `Agent/`.
- Use `uv` for Python environment management and Python script execution; prefer `uv run python ...` or `uv run <script>` over bare `python`/`python3`.
- Strong correction from the current collaboration: for the handwritten HMX MatMul roadmap, keep the main line on the owned QNN-free runtime under `example/handwritten_hmx_matmul/`, specifically the tutorial/direct-HMX wrapper route in `example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/`. QNN Native/custom-op/descdump work is only an offline artifact/oracle source; do not continue QNN context/preheader/prepad/mutation blackbox probing as the implementation goal.
- For W4A16 handwritten work, do not run broad wrapper/HMX-environment sweeps; derive one named `cvt.uh`, folded-bias record, accumulator-packing, or code-identified pre-function state hypothesis before a device probe.
- For the active handwritten route, use `tests/handwritten_hmx_matmul/run_all.sh` as the route gate; it must stay on tutorial/direct-HMX checks and must not wire old W4A16 blackbox matrices.
- Before every formal code commit or push, run the full kernel E2E CI gate: `tests/qnn_kernel_e2e/run_all.sh`.
- Enable the repo-local pre-push hook with `scripts/install_git_hooks.sh` in each checkout.
- Do not add new agent docs under repo root, `docs/`, or ad-hoc folders.
- Keep non-Markdown reverse-engineering evidence under `Agent/qnn_re/`.
