# Agent.md

`Agent/` is the only home for agent-managed project knowledge in this repo.

Start here: [Agent/README.md](Agent/README.md).

Rules:
- Keep this file short.
- Keep `Agent/README.md` as a short link-oriented entrypoint.
- Put current status, handoffs, findings, and project-specific rules in linked documents under `Agent/`.
- Use `uv` for Python environment management and Python script execution; prefer `uv run python ...` or `uv run <script>` over bare `python`/`python3`.
- Before every formal code commit or push, run the full kernel E2E CI gate: `tests/qnn_kernel_e2e/run_all.sh`.
- Enable the repo-local pre-push hook with `scripts/install_git_hooks.sh` in each checkout.
- Do not add new agent docs under repo root, `docs/`, or ad-hoc folders.
- Keep non-Markdown reverse-engineering evidence under `Agent/qnn_re/`.
