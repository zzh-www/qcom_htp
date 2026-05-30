#!/usr/bin/env python3
"""Extract the frozen golden I/O for the Qwen3.5 Gated-DeltaNet (GDN) core kernel.

Boundary captured = the exact `chunk_gated_delta_rule(query, key, value, g=, beta=, ...)`
call inside `Qwen3_5GatedDeltaNet` — the unit we port to HTP. For each prompt in the
frozen corpus (tests/gdn/prompts.json) we run one bf16 forward pass and dump, per GDN
layer, the kernel inputs (query,key,value,g,beta) and the model output (o). A single
capture file is one (prompt, layer); calibration/evaluation is done PER LAYER downstream.

Tensors are stored LOSSLESSLY as bf16 bit patterns (uint16) to keep the corpus compact
and exactly faithful to what the model feeds the kernel. The fp32 reference output is NOT
stored here — it is derived from these inputs by scripts/gdn_ref_kernel.gdn_kernel (which
is validated bit-for-bit against the FLA reference to ~2e-7).

Usage:
  env -u http_proxy -u https_proxy ... .venv/bin/python scripts/gdn_extract_golden.py \
      --model downloads/Qwen3.5-4B --prompts tests/gdn/prompts.json --out tests/gdn/golden
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch


def load_model(model_path: str, dtype):
    import transformers
    last_err = None
    for loader_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        if not hasattr(transformers, loader_name):
            continue
        Loader = getattr(transformers, loader_name)
        for dtype_kw in ("dtype", "torch_dtype"):
            try:
                model = Loader.from_pretrained(
                    model_path, trust_remote_code=True, device_map="cuda",
                    low_cpu_mem_usage=True, **{dtype_kw: dtype})
                print(f"[load] via {loader_name} ({dtype_kw}): {type(model).__name__}")
                return model
            except TypeError as e:
                last_err = e
            except Exception as e:  # noqa: BLE001
                last_err = e; print(f"[load] {loader_name} failed: {type(e).__name__}: {e}"); break
    raise RuntimeError(f"could not load model: {last_err}")


def find_gdn_modules(model):
    out = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "Qwen3_5GatedDeltaNet" and hasattr(mod, "chunk_gated_delta_rule"):
            out.append((name, mod))
    return out


def bf16_bits(t):
    """Store a bf16 tensor losslessly as its uint16 bit pattern; else fp32."""
    if t is None:
        return None
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).cpu().numpy().view(np.uint16)
    return t.float().cpu().numpy()


def layer_index(name):
    for p in name.split("."):
        if p.isdigit():
            return int(p)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="downloads/Qwen3.5-4B")
    ap.add_argument("--prompts", default="tests/gdn/prompts.json")
    ap.add_argument("--out", default="tests/gdn/golden")
    args = ap.parse_args()

    torch.manual_seed(0)
    corpus = json.load(open(args.prompts))
    prompts = corpus["prompts"]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    print(f"[info] transformers {__import__('transformers').__version__}, torch {torch.__version__}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, torch.bfloat16); model.eval()
    gdn = find_gdn_modules(model)
    print(f"[info] {len(gdn)} GDN layers, {len(prompts)} prompts")

    records = []
    cur = {"pid": None, "split": None}

    def make_wrapper(orig, lid):
        def wrapper(query, key, value, **kw):
            out = orig(query, key, value, **kw)
            core_attn_out, last_state = out
            fname = f"p{cur['pid']:02d}_L{lid:02d}.npz"
            np.savez(out_dir / fname,
                     query=bf16_bits(query), key=bf16_bits(key), value=bf16_bits(value),
                     g=bf16_bits(kw.get("g")), beta=bf16_bits(kw.get("beta")),
                     o=bf16_bits(core_attn_out))
            records.append(dict(
                file=fname, prompt_id=cur["pid"], split=cur["split"], layer=lid,
                T=int(query.shape[1]), n_chunks=(int(query.shape[1]) + 63) // 64,
                shapes=dict(query=list(query.shape), value=list(value.shape),
                            g=list(kw["g"].shape), o=list(core_attn_out.shape)),
                dtype="bf16-bits-uint16",
                use_qk_l2norm=bool(kw.get("use_qk_l2norm_in_kernel")),
            ))
            return out
        return wrapper

    for name, mod in gdn:
        mod.chunk_gated_delta_rule = make_wrapper(mod.chunk_gated_delta_rule, layer_index(name))

    for p in prompts:
        cur["pid"], cur["split"] = p["id"], p["split"]
        ids = tok(p["text"], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            model(**ids, use_cache=False)
        print(f"[run] p{p['id']:02d} {p['split']:5s} len={ids['input_ids'].shape[1]} "
              f"caps={len(records)} ({time.time()-t0:.1f}s)")

    (out_dir / "manifest.json").write_text(json.dumps(dict(
        model=args.model, prompts_file=args.prompts, prompts_version=corpus.get("version"),
        chunk_size=corpus.get("chunk_size", 64),
        transformers=__import__("transformers").__version__, torch=torch.__version__,
        n_prompts=len(prompts), n_captures=len(records),
        n_calib=sum(1 for p in prompts if p["split"] == "calib"),
        n_test=sum(1 for p in prompts if p["split"] == "test"),
        records=records,
    ), indent=2, ensure_ascii=False))
    print(f"[done] {len(records)} captures -> {out_dir}/  (manifest.json)")


if __name__ == "__main__":
    main()
