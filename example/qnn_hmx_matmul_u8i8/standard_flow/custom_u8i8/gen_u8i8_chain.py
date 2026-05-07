#!/usr/bin/env python3
"""
Generate the custom_u8i8 ONNX test graph.

The only custom op emitted here is HmxU8I8ToU8MatMul:
  bias      [1, N/32, 1, 64]      int32 initializer, native folded bytes
  weight    [1, 1, K, N]          uint8 initializer, bytes are K-major HMX tiles
  activation[1, M/32, 32, K]      uint8 model input, converted to Crouton_8 by QNN
  scratch   [1, 1, 1, 2048]       uint8 initializer
  output    [1, M/32, 32, N]      uint8 Crouton_8, reshaped by QNN to [1,1,M,N]
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "hmx"
ACT_ZP = 128


def pack_weight_kmajor(w_raw_kn: np.ndarray) -> np.ndarray:
    """Pack logical [K, N] int8 weights into the V73DEEP HMX tile stream.

    The runtime wrapper passes the resulting bytes directly as r2/wt_pack.  No
    device-side repack is expected.  Each 32x32 logical tile is stored as 1024
    bytes using the lane order decoded from QNN's Conv1x1 weight layout:

        dst = (k_row / 4) * 128 + n_col * 4 + (k_row % 4)

    The outer tile order is K-major, then N:

        [K tile 0, N tile 0], [K tile 0, N tile 1], ...
        [K tile 1, N tile 0], [K tile 1, N tile 1], ...
    """
    k, n = w_raw_kn.shape
    k_t, n_t = k // 32, n // 32
    packed = np.zeros((1, k_t, n_t, 1024), dtype=np.int8)
    for kt in range(k_t):
        for nt in range(n_t):
            for r in range(32):
                for c in range(32):
                    dst = (r // 4) * 128 + c * 4 + (r % 4)
                    packed[0, kt, nt, dst] = w_raw_kn[kt * 32 + r, nt * 32 + c]
    return packed


def make_activation(m: int, k: int, idx: int) -> np.ndarray:
    seed = (idx + 1) * 374761393
    return np.array(
        [((i * 37 + seed) & 0xFF) for i in range(m * k)],
        dtype=np.uint8,
    ).reshape(1, 1, m, k)


def make_reference(a_raw: np.ndarray, w_raw_kn: np.ndarray, bias_q: np.ndarray) -> np.ndarray:
    acc = (a_raw.reshape(w_raw_kn.shape[0], w_raw_kn.shape[0]).astype(np.int32) - ACT_ZP)
    out = acc @ w_raw_kn.astype(np.int32) + bias_q
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seed", type=int, default=0xB17E)
    p.add_argument("--chain", type=int, default=8)
    p.add_argument("--mode", choices=["chain", "independent"], default="chain")
    p.add_argument("-o", "--out", default="u8i8_chain.onnx")
    args = p.parse_args()

    assert args.M == args.K == args.N, "current replica flow is square-only"
    assert args.M % 32 == 0 and args.K % 32 == 0 and args.N % 32 == 0

    m, k, n = args.M, args.K, args.N
    k_t, n_t = k // 32, n // 32
    chain = max(1, int(args.chain))
    np.random.seed(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    k_idx, n_idx = np.meshgrid(np.arange(k), np.arange(n), indexing="ij")
    w_raw_kn = (((k_idx * 31 + n_idx * 13) % 15) - 7).astype(np.int8)
    wt_packed = pack_weight_kmajor(w_raw_kn)
    # The ONNX tensor keeps a native-looking [1, 1, K, N] shape so converter and
    # ctxgen accept it as a normal weight initializer.  Its bytes are already in
    # the HMX K-major tile order, so the runtime can pass qhpi_tensor_raw_data()
    # straight to our_v73deep_kernel().
    wt_init = numpy_helper.from_array(
        wt_packed.view(np.uint8).reshape(1, 1, k, n),
        name="weight",
    )

    # Fold the activation zero point into the bias offline:
    #
    #   (act_u8 - ACT_ZP) @ weight + bias_q
    #     = act_u8 @ weight + (-ACT_ZP * sum_k(weight) + bias_q)
    #
    # The V73DEEP Conv1x1 body consumes a 256-byte native bias record per N tile:
    # first 128 bytes are 32 fp16 scale/baseline pairs, second 128 bytes are
    # 32 int32 effective-bias values.  Runtime therefore only forwards a pointer.
    bias_q = np.arange(1, n + 1, dtype=np.int32)
    sum_w = w_raw_kn.astype(np.int32).sum(axis=0)
    effective_i32 = (-ACT_ZP) * sum_w + bias_q

    bias_fold_bytes = np.zeros((n_t, 256), dtype=np.uint8)
    for nt in range(n_t):
        for c in range(32):
            col = nt * 32 + c
            scale_u16 = np.float16(512.0).view(np.uint16).item()
            baseline_u16 = 0
            bias_fold_bytes[nt, 4 * c:4 * c + 2] = np.array([scale_u16], np.uint16).view(np.uint8)
            bias_fold_bytes[nt, 4 * c + 2:4 * c + 4] = np.array([baseline_u16], np.uint16).view(np.uint8)
            bias_fold_bytes[nt, 128 + 4 * c:128 + 4 * c + 4] = (
                np.array([int(effective_i32[col])], np.int32).view(np.uint8)
            )
    bias_init = numpy_helper.from_array(
        bias_fold_bytes.view(np.int32).reshape(1, n_t, 1, 64).copy(),
        name="bias",
    )

    scratch_init = numpy_helper.from_array(
        np.zeros((1, 1, 1, 2048), dtype=np.uint8),
        name="scratch",
    )
    # QNN will materialize the custom op activation/output as Crouton_8 indirect
    # tensors.  The surrounding Reshape nodes keep the public graph in ordinary
    # [1, 1, M, K]/[1, 1, M, N] form while giving the custom op the 4D shape its
    # QHPI signature expects: [1, M/32, 32, K].
    in_reshape_dims = numpy_helper.from_array(
        np.array([1, m // 32, 32, k], dtype=np.int64),
        name="in_reshape_dims",
    )
    out_reshape_dims = numpy_helper.from_array(
        np.array([1, 1, m, n], dtype=np.int64),
        name="out_reshape_dims",
    )

    initializers = [wt_init, bias_init, scratch_init, in_reshape_dims, out_reshape_dims]
    inputs_info = []
    outputs_info = []
    value_infos = []
    nodes = []

    if args.mode == "chain":
        inputs_info.append(helper.make_tensor_value_info("act_raw", TensorProto.UINT8, [1, 1, m, k]))
        outputs_info.append(helper.make_tensor_value_info("out", TensorProto.UINT8, [1, 1, m, n]))
        nodes.append(helper.make_node("Reshape", ["act_raw", "in_reshape_dims"], ["act_4d"], name="reshape_in"))
        prev = "act_4d"
        for i in range(chain):
            out_name = f"hmx_u8i8_{i}" if i < chain - 1 else "hmx_u8i8"
            nodes.append(helper.make_node(
                "HmxU8I8ToU8MatMul",
                inputs=["bias", "weight", prev, "scratch"],
                outputs=[out_name],
                name=f"hmx_u8i8_chain{i}",
                domain=DOMAIN,
            ))
            value_infos.append(helper.make_tensor_value_info(out_name, TensorProto.UINT8, [1, m // 32, 32, n]))
            prev = out_name
        nodes.append(helper.make_node("Reshape", ["hmx_u8i8", "out_reshape_dims"], ["out"], name="reshape_out"))
    else:
        for i in range(chain):
            in_name = "act_raw" if i == 0 else f"act_raw_{i}"
            out_name = f"out_{i}"
            act4d_name = f"act_4d_{i}"
            mm_name = f"hmx_u8i8_{i}"
            in_shape_name = "in_reshape_dims" if i == 0 else f"in_reshape_dims_{i}"
            out_shape_name = "out_reshape_dims" if i == 0 else f"out_reshape_dims_{i}"
            if i > 0:
                initializers.append(numpy_helper.from_array(
                    np.array([1, m // 32, 32, k], dtype=np.int64),
                    name=in_shape_name,
                ))
                initializers.append(numpy_helper.from_array(
                    np.array([1, 1, m, n], dtype=np.int64),
                    name=out_shape_name,
                ))
            inputs_info.append(helper.make_tensor_value_info(in_name, TensorProto.UINT8, [1, 1, m, k]))
            outputs_info.append(helper.make_tensor_value_info(out_name, TensorProto.UINT8, [1, 1, m, n]))
            nodes.append(helper.make_node("Reshape", [in_name, in_shape_name], [act4d_name], name=f"reshape_in_{i}"))
            nodes.append(helper.make_node(
                "HmxU8I8ToU8MatMul",
                inputs=["bias", "weight", act4d_name, "scratch"],
                outputs=[mm_name],
                name=f"hmx_u8i8_indep{i}",
                domain=DOMAIN,
            ))
            value_infos.append(helper.make_tensor_value_info(mm_name, TensorProto.UINT8, [1, m // 32, 32, n]))
            nodes.append(helper.make_node("Reshape", [mm_name, out_shape_name], [out_name], name=f"reshape_out_{i}"))

    graph = helper.make_graph(
        nodes,
        name=f"custom_u8i8_{args.mode}",
        inputs=inputs_info,
        outputs=outputs_info,
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name=f"custom_u8i8_{m}x{k}x{n}",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
    )
    model.ir_version = 8
    onnx.save(model, out_path)

    u8_enc = {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "False",
        "scale": 1.0,
        "offset": 0,
        "min": 0.0,
        "max": 255.0,
    }
    bias_enc = {
        "bitwidth": 32,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0,
        "offset": 0,
        "min": -2147483648.0,
        "max": 2147483647.0,
    }
    if args.mode == "chain":
        activation_encodings = {"act_raw": [u8_enc], "out": [u8_enc]}
    else:
        activation_encodings = {"act_raw": [u8_enc]}
        for i in range(chain):
            if i > 0:
                activation_encodings[f"act_raw_{i}"] = [u8_enc]
            activation_encodings[f"out_{i}"] = [u8_enc]
    overrides = {
        "activation_encodings": activation_encodings,
        "param_encodings": {"bias": [bias_enc]},
    }
    with open(os.path.join(os.path.dirname(out_path), "quant_overrides.json"), "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)

    out_dir = os.path.dirname(out_path)
    runtime_dir = os.path.join(out_dir, "runtime_inputs_u8")
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_input_list = os.path.join(out_dir, "runtime_input_list.txt")
    if args.mode == "chain":
        a0 = make_activation(m, k, 0)
        a0.tofile(os.path.join(runtime_dir, "act_u8i8.raw"))
        with open(runtime_input_list, "w", encoding="utf-8") as f:
            f.write("act_raw:=runtime_inputs_u8/act_u8i8.raw\n")
        cur = a0.reshape(m, k).astype(np.int32)
        for _ in range(chain):
            cur = np.clip((cur - ACT_ZP) @ w_raw_kn.astype(np.int32) + bias_q, 0, 255)
        out_ref = cur.astype(np.uint8)
    else:
        input_parts = []
        for i in range(chain):
            a_i = make_activation(m, k, i)
            fname = "act_u8i8.raw" if i == 0 else f"act_u8i8_{i}.raw"
            a_i.tofile(os.path.join(runtime_dir, fname))
            name = "act_raw" if i == 0 else f"act_raw_{i}"
            input_parts.append(f"{name}:=runtime_inputs_u8/{fname}")
        with open(runtime_input_list, "w", encoding="utf-8") as f:
            f.write(" ".join(input_parts) + "\n")
        out_ref = make_reference(make_activation(m, k, 0), w_raw_kn, bias_q)

    with open(os.path.join(out_dir, "native_io.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_name": "act_raw" if args.mode == "chain" else [
                    "act_raw" if i == 0 else f"act_raw_{i}" for i in range(chain)
                ],
                "output_name": "out" if args.mode == "chain" else [f"out_{i}" for i in range(chain)],
                "native_input": "runtime_inputs_u8/act_u8i8.raw",
                "runtime_input_list": "runtime_input_list.txt",
                "native_input_storage": "uint8",
                "native_input_bytes": int(m * k),
                "expected_native_output_storage": "uint8",
                "expected_native_output_bytes": int(m * n),
                "shape_mkn": [m, k, n],
                "graph_input_shape": [1, 1, m, k],
                "graph_output_shape": [1, 1, m, n],
            },
            f,
            indent=2,
        )

    np.save(out_path + ".wRaw_KN.npy", w_raw_kn)
    np.save(out_path + ".bias_q_int32.npy", bias_q)
    np.save(out_path + ".effective_int32.npy", effective_i32)
    np.save(out_path + ".out_ref_u8.npy", out_ref)

    print(f"  -> {out_path}")
    print(f"  graph: HmxU8I8ToU8MatMul x {chain} ({args.mode})")
    print(f"  shape: M={m} K={k} N={n}; wt=K-major; ACT_ZP={ACT_ZP}")
    print(f"  ref out[0..3,0]: {out_ref[:4, 0].tolist()}")


if __name__ == "__main__":
    main()
