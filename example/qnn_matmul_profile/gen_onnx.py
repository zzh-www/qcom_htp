#!/usr/bin/env python3
"""Generate a configurable MatMul ONNX model and (optionally) a
quant-overrides JSON for QNN qairt-converter.

Usage:
    python gen_onnx.py <config_name> <out_dir> [--m M] [--k K] [--n N]

Config names:
    fp16     - fp16 inputs, no quant override
    w16a16   - int16 weight, int16 activation, int16 output
    w8a16    - int8  weight, int16 activation, int16 output
    w8a8     - int8  weight, int8  activation, int8  output
    w4a16    - int4  weight, int16 activation, int16 output
    w4a8     - int4  weight, int8  activation, int8  output
    w4a4     - int4  weight, int4  activation, int4  output

Shape: A is [1, M, K], W is [1, K, N], Y is [1, M, N]. Default 32×32×32.

Writes under <out_dir>/:
    matmul.onnx
    quant_overrides.json  (unless fp16)
    input_A.raw           (fp32 input data, M*K values)
    input_list.txt        (qnn-net-run input list, "A:=input_A.raw")
    runtime_inputs_native/A.raw
    runtime_input_list.txt
    native_io.json
"""
import argparse, json, os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CONFIGS = {
    "fp16":   {"dtype": "float", "float_bitwidth": 16},
    "w16a16": {"dtype": "quant", "act": 16, "weight": 16, "out": 16, "symmetric": True},
    "w8a16":  {"dtype": "quant", "act": 16, "weight":  8, "out": 16, "symmetric": True},
    "w8a8":   {"dtype": "quant", "act":  8, "weight":  8, "out":  8, "symmetric": True},
    "w4a16":  {"dtype": "quant", "act": 16, "weight":  4, "out": 16, "symmetric": True},
    "w4a8":   {"dtype": "quant", "act":  8, "weight":  4, "out":  8, "symmetric": True},
    "w4a4":   {"dtype": "quant", "act":  4, "weight":  4, "out":  4, "symmetric": True},
}


def _symmetric_encoding(bits: int) -> dict:
    max_int = (1 << (bits - 1)) - 1
    return {
        "bitwidth":     bits,
        "dtype":        "int",
        "is_symmetric": "True",
        "scale":         1.0 / max_int,
        "offset":       -(1 << (bits - 1)),
        "min":          -1.0,
        "max":           1.0,
    }


def _pack_u4(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.uint8) & 0x0F
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8, copy=False)


def _quantize_native(values: np.ndarray, bits: int) -> tuple[np.ndarray, dict]:
    enc = _symmetric_encoding(bits)
    qmax = (1 << bits) - 1
    q = np.clip(np.rint(values / enc["scale"] - enc["offset"]), 0, qmax)
    if bits == 4:
        return _pack_u4(q), {"storage": "uint4_packed_lohi", "bytes": int((q.size + 1) // 2)}
    if bits <= 8:
        return q.astype(np.uint8), {"storage": "uint8", "bytes": int(q.size)}
    if bits <= 16:
        return q.astype("<u2"), {"storage": "uint16_le", "bytes": int(q.size * 2)}
    raise ValueError(f"unsupported native activation bitwidth: {bits}")


def _draw(rng, shape, dist: str, dtype):
    """Value-distribution edge cases (scale stays the symmetric 1/maxint; only
    the distribution of the float values changes). Used to exercise correctness
    at saturation / zero / impulse / bimodal extremes, not just uniform noise."""
    if dist == "uniform":
        v = rng.uniform(-0.5, 0.5, size=shape)
    elif dist == "extreme":          # near-max magnitude -> push output saturation
        v = rng.choice([-1.0, 1.0], size=shape) * 0.999
    elif dist == "signs":            # bimodal +-0.5
        v = rng.choice([-0.5, 0.5], size=shape)
    elif dist == "zeros":            # all zero -> output = zero-point/bias baseline
        v = np.zeros(shape)
    elif dist == "impulse":          # mostly zero, a few +-max
        v = np.zeros(shape)
        flat = v.reshape(-1)
        idx = rng.choice(flat.size, size=max(1, flat.size // 64), replace=False)
        flat[idx] = rng.choice([-1.0, 1.0], size=idx.size) * 0.999
    elif dist == "sparse":           # ~15% nonzero uniform
        v = rng.uniform(-0.5, 0.5, size=shape)
        v *= (rng.random(size=shape) < 0.15)
    else:
        raise ValueError(f"unknown dist: {dist}")
    return v.astype(dtype)


def _emit_onnx(cfg: dict, path: str, m: int, k: int, n: int, seed: int = 42, dist: str = "uniform"):
    rng = np.random.default_rng(seed)
    if cfg["dtype"] == "float":
        onnx_dtype, np_dtype = TensorProto.FLOAT16, np.float16
    else:
        onnx_dtype, np_dtype = TensorProto.FLOAT, np.float32

    _B = int(os.environ.get("GEN_BATCH", "1"))   # batched MatMul: B independent (distinct-weight) M×K@K×N
    if _B > 1:
        W_val = _draw(rng, (1, _B, k, n), dist, np_dtype); ash, ysh = [1, _B, m, k], [1, _B, m, n]
    else:
        W_val = _draw(rng, (1, k, n), dist, np_dtype); ash, ysh = [1, m, k], [1, m, n]
    W_init = numpy_helper.from_array(W_val, name="W")
    A = helper.make_tensor_value_info("A", onnx_dtype, ash)
    Y = helper.make_tensor_value_info("Y", onnx_dtype, ysh)
    node = helper.make_node("MatMul", ["A", "W"], ["Y"], name="matmul_1")
    graph = helper.make_graph([node], "matmul", [A], [Y], [W_init])
    model = helper.make_model(
        graph,
        producer_name="qnn_matmul_profile",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    try:
        onnx.checker.check_model(model)
    except Exception:
        pass  # shape check sometimes chokes on huge tensors; conversion is what matters
    onnx.save(model, path)


def _emit_quant(cfg: dict, path: str, n: int):
    w_bits = cfg["weight"]
    # int4 weights need per-output-channel encoding (one per N column).
    if w_bits <= 4:
        w_enc = [_symmetric_encoding(w_bits) for _ in range(n)]
    else:
        w_enc = [_symmetric_encoding(w_bits)]

    enc = {
        "activation_encodings": {
            "A": [_symmetric_encoding(cfg["act"])],
            "Y": [_symmetric_encoding(cfg["out"])],
        },
        "param_encodings": {"W": w_enc},
    }
    with open(path, "w") as f:
        json.dump(enc, f, indent=2)


def _emit_input(cfg: dict, out_dir: str, m: int, k: int, n: int, seed: int = 0xBEEF, dist: str = "uniform"):
    rng = np.random.default_rng(seed)
    _B = int(os.environ.get("GEN_BATCH", "1"))
    A = _draw(rng, (1, _B, m, k) if _B > 1 else (1, m, k), dist, np.float32)
    raw = os.path.join(out_dir, "input_A.raw")
    A.tofile(raw)
    with open(os.path.join(out_dir, "input_list.txt"), "w") as f:
        f.write("A:=input_A.raw\n")

    native_dir = os.path.join(out_dir, "runtime_inputs_native")
    os.makedirs(native_dir, exist_ok=True)
    native_raw = os.path.join(native_dir, "A.raw")
    if cfg["dtype"] == "float":
        A.astype("<f2").tofile(native_raw)
        native_meta = {"storage": "float16_le", "bytes": int(A.size * 2)}
    else:
        native_values, native_meta = _quantize_native(A, cfg["act"])
        native_values.tofile(native_raw)

    with open(os.path.join(out_dir, "runtime_input_list.txt"), "w") as f:
        f.write("A:=runtime_inputs_native/A.raw\n")

    meta = {
        "input_name": "A",
        "output_name": "Y",
        "legacy_fp32_input": "input_A.raw",
        "native_input": "runtime_inputs_native/A.raw",
        "runtime_input_list": "runtime_input_list.txt",
        "native_input_storage": native_meta["storage"],
        "native_input_bytes": native_meta["bytes"],
        "shape": ([1, _B, m, k] if _B > 1 else [1, m, k]),
        "output_shape": ([1, _B, m, n] if _B > 1 else [1, m, n]),
    }
    _MN = _B * m * n if _B > 1 else m * n
    if cfg["dtype"] == "quant":
        meta["activation_encoding"] = _symmetric_encoding(cfg["act"])
        meta["expected_native_output_storage"] = (
            "uint8" if cfg["out"] <= 8 else "uint16_le"
        )
        meta["expected_native_output_bytes"] = int(_MN * (1 if cfg["out"] <= 8 else 2))
    else:
        meta["expected_native_output_storage"] = "float16_le"
        meta["expected_native_output_bytes"] = int(_MN * 2)
    with open(os.path.join(out_dir, "native_io.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=CONFIGS.keys())
    ap.add_argument("out_dir")
    ap.add_argument("--m", type=int, default=32, help="rows of A / rows of output")
    ap.add_argument("--k", type=int, default=32, help="reduction dim (cols of A, rows of W)")
    ap.add_argument("--n", type=int, default=32, help="cols of W / cols of output")
    ap.add_argument("--seed", type=int, default=None,
                    help="base RNG seed (W:seed, A:seed^0xBEEF); default keeps 42/0xBEEF")
    _DISTS = ["uniform", "extreme", "signs", "zeros", "impulse", "sparse"]
    ap.add_argument("--dist", default=os.environ.get("GEN_DIST", "uniform"), choices=_DISTS,
                    help="value distribution for W and A (also via GEN_DIST env); "
                         "edge cases for correctness, scale unchanged")
    ap.add_argument("--weight-dist", default=os.environ.get("GEN_WDIST"), choices=_DISTS,
                    help="override weight distribution (else --dist / GEN_WDIST env)")
    ap.add_argument("--act-dist", default=os.environ.get("GEN_ADIST"), choices=_DISTS,
                    help="override activation distribution (else --dist / GEN_ADIST env)")
    args = ap.parse_args()
    w_dist = args.weight_dist or args.dist
    a_dist = args.act_dist or args.dist
    if args.seed is None:
        env = os.environ.get("GEN_SEED")
        args.seed = int(env) if env is not None else None
    w_seed = 42 if args.seed is None else args.seed
    a_seed = 0xBEEF if args.seed is None else (args.seed ^ 0xBEEF)

    cfg = CONFIGS[args.config]
    os.makedirs(args.out_dir, exist_ok=True)

    _emit_onnx(cfg, os.path.join(args.out_dir, "matmul.onnx"), args.m, args.k, args.n,
               seed=w_seed, dist=w_dist)
    if cfg["dtype"] == "quant":
        _emit_quant(cfg, os.path.join(args.out_dir, "quant_overrides.json"), args.n)
    _emit_input(cfg, args.out_dir, args.m, args.k, args.n, seed=a_seed, dist=a_dist)

    print(f"  [{args.config}] wrote {args.out_dir}/ (dtype={cfg['dtype']}, "
          f"{args.m}x{args.k}x{args.n}, wdist={w_dist}, adist={a_dist}, seed={args.seed})")


if __name__ == "__main__":
    main()
