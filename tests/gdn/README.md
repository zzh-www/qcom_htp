# GDN kernel tests

Single-layer correctness tests for the Gated DeltaNet (GDN) linear-attention kernel being
ported to Hexagon HTP, validated against the real `Qwen/Qwen3.5-4B` model.

**Design baseline & full context:** [`Agent/current/gdn_kernel_reference.md`](../../Agent/current/gdn_kernel_reference.md).

- `prompts.json` — frozen prompt corpus (append-only: never edit existing ids/text).
- `test_gdn_layer.py` — **one test == one GDN layer**: static per-layer w8a16 scales built
  from the layer's calib prompts, evaluated on held-out test prompts vs the fp32 reference.
- `golden/` — per-(prompt,layer) kernel I/O, gitignored. Regenerate:

```bash
env -u http_proxy -u https_proxy -u all_proxy .venv/bin/python scripts/gdn_extract_golden.py \
    --model downloads/Qwen3.5-4B --prompts tests/gdn/prompts.json --out tests/gdn/golden
```

Run the tests:

```bash
.venv/bin/python tests/gdn/test_gdn_layer.py          # default: w8a16, focus L00
GDN_LAYER=all .venv/bin/python tests/gdn/test_gdn_layer.py
.venv/bin/python -m pytest tests/gdn/test_gdn_layer.py -q
```
