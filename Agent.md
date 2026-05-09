# Agent.md

`Agent/` is the only home for agent-managed project knowledge in this repo.

Start here: [Agent/README.md](Agent/README.md).

Rules:
- Keep this file short.
- Keep `Agent/README.md` as a short link-oriented entrypoint.
- Put current status, handoffs, findings, and project-specific rules in linked documents under `Agent/`.
- Use `uv` for Python environment management and Python script execution; prefer `uv run python ...` or `uv run <script>` over bare `python`/`python3`.
- Do not add new agent docs under repo root, `docs/`, or ad-hoc folders.
- Keep non-Markdown reverse-engineering evidence under `Agent/qnn_re/`.
