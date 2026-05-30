#!/usr/bin/env python3
"""Run a generated Python-reference MatMul case through QNN native HTP.

The input case is produced by ``generate_qnn_matmul_python_cases.py``.  This
runner builds a native Conv1x1 graph equivalent to ``X @ W.T + bias`` so weight
per-output-channel scales map to Conv axis 0.  Device output is compared against
the case's Python quantized reference.  The reference uses AIMET/QNN affine QDQ
semantics.  Only this QNN Native HTP vs Python/AIMET-style oracle comparison
allows small integer-output deltas.  Same-hardware comparisons, including custom
op vs native op and handwritten vs custom op, remain exact-output gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


W4A16_ACCEPTED_MAXDIFF_BY_CASE = {
    "normal_random": 6,
    "single_k_impulse": 36,
    "scale_only": 3,
}


def accepted_maxdiff_for_case(meta: dict[str, Any], override: int | None = None) -> int:
    if override is not None:
        return override
    if meta.get("family") in {"w4a16_per_channel", "w4a16_lpbq"}:
        return W4A16_ACCEPTED_MAXDIFF_BY_CASE.get(str(meta.get("case")), 1)
    return 1


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def ssh_write(device: str, remote_path: str, local_path: Path) -> None:
    with local_path.open("rb") as f:
        subprocess.run(["ssh", device, f"cat > {remote_path}"], stdin=f, check=True)


def ssh_read(device: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as f:
        subprocess.run(["ssh", device, f"cat {remote_path}"], stdout=f, check=True)


def load_case(case_dir: Path) -> dict[str, Any]:
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    arrays = {
        "activation_float": np.load(case_dir / meta["files"]["activation_float"]["npy"]),
        "activation_q": np.load(case_dir / meta["files"]["activation_q"]["npy"]),
        "weight_float_nk": np.load(case_dir / meta["files"]["weight_float_nk"]["npy"]),
        "weight_q_nk": np.load(case_dir / meta["files"]["weight_q_nk"]["npy"]),
        "weight_scale": np.load(case_dir / meta["files"]["weight_scale"]["npy"]),
        "bias_float": np.load(case_dir / meta["files"]["bias_float"]["npy"]),
        "output_ref_q": np.load(case_dir / meta["files"]["output_ref_q"]["npy"]),
    }
    if "weight_lpbq_per_block_int_scale" in meta.get("files", {}):
        arrays["weight_lpbq_per_block_int_scale"] = np.load(
            case_dir / meta["files"]["weight_lpbq_per_block_int_scale"]["npy"]
        )
    return {"meta": meta, "arrays": arrays}


def dequant_weight(weight_q: np.ndarray, weight_scale: np.ndarray) -> np.ndarray:
    if weight_scale.ndim == 1:
        return weight_q.astype(np.float32) * weight_scale[:, None].astype(np.float32)
    if weight_scale.ndim == 2:
        n, k = weight_q.shape
        groups = weight_scale.shape[1]
        if k % groups:
            raise ValueError(f"weight K={k} is not divisible by scale groups={groups}")
        group_size = k // groups
        out = np.empty((n, k), dtype=np.float32)
        for group in range(groups):
            start = group * group_size
            stop = start + group_size
            out[:, start:stop] = weight_q[:, start:stop].astype(np.float32) * weight_scale[:, group, None]
        return out
    raise ValueError(f"unsupported weight_scale rank: {weight_scale.ndim}")


def enc_v1(name: str, bitwidth: int, scale: float, zp: int, *, symmetric: bool) -> dict[str, Any]:
    qmax = (1 << bitwidth) - 1
    qmin = 0
    return {
        "name": name,
        "enc_type": "PER_TENSOR",
        "bw": bitwidth,
        "dtype": "INT",
        "is_sym": bool(symmetric),
        "scale": [float(scale)],
        "offset": [-int(zp)],
        "min": [(qmin - int(zp)) * float(scale)],
        "max": [(qmax - int(zp)) * float(scale)],
    }


def weight_enc_v1(name: str, bitwidth: int, scales: np.ndarray) -> dict[str, Any]:
    flat = scales.astype(np.float64).reshape(-1).tolist()
    qmax = (1 << (bitwidth - 1)) - 1
    offset = -(1 << (bitwidth - 1)) if bitwidth == 4 else 0
    qmin = -qmax - 1 if bitwidth == 4 else -qmax
    if len(flat) == 1:
        enc_type = "PER_TENSOR"
    else:
        enc_type = "PER_CHANNEL"
    return {
        "name": name,
        "enc_type": enc_type,
        "bw": bitwidth,
        "dtype": "INT",
        "is_sym": True,
        "scale": flat,
        "offset": [offset] * len(flat),
        "min": [qmin * s for s in flat],
        "max": [qmax * s for s in flat],
        "axis": 0,
    }


def lpbq_weight_enc_v1(
    name: str,
    scales: np.ndarray,
    per_block_int_scale: np.ndarray,
    block_size: int,
) -> dict[str, Any]:
    flat_scales = scales.astype(np.float64).reshape(-1).tolist()
    per_block = per_block_int_scale.astype(np.uint8)
    if per_block.ndim != 2:
        raise ValueError(f"LPBQ per_block_int_scale must be rank-2, got {per_block.shape}")
    if per_block.shape[0] != len(flat_scales):
        raise ValueError(
            "LPBQ per_block_int_scale channel count mismatch: "
            f"{per_block.shape[0]} != {len(flat_scales)}"
        )
    return {
        "name": name,
        "enc_type": "LPBQ",
        "bw": 8,
        "dtype": "INT",
        "is_sym": True,
        "compressed_bw": 4,
        "block_size": int(block_size),
        "scale": flat_scales,
        "offset": [-128] * len(flat_scales),
        "per_block_int_scale": per_block.astype(int).tolist(),
    }


def write_artifact(case_dir: Path, out_dir: Path, w4_encoding: str) -> dict[str, Any]:
    case = load_case(case_dir)
    meta = case["meta"]
    arr = case["arrays"]
    family = meta["family"]
    qparams = meta["qparams"]
    m, k, n = meta["shape_mkn"]
    if meta.get("weight_schema_variant") not in {"per_output_channel", "lpbq_blockwise_expansion"}:
        raise ValueError("native Conv1x1 runner currently supports only per-output-channel or LPBQ weight scales")
    if family not in {"u8i8", "w4a8_per_channel", "w4a8_lpbq", "w8a16", "w4a16_per_channel", "w4a16_lpbq"}:
        raise ValueError(f"unsupported native Conv1x1 family: {family}")
    if w4_encoding == "lpbq":
        if family not in {"w4a8_lpbq", "w4a16_lpbq"}:
            raise ValueError(f"--w4-encoding lpbq requires an LPBQ case family, got {family}")
        if "weight_lpbq_per_block_int_scale" not in arr:
            raise ValueError(f"{case_dir}: missing LPBQ per-block scale artifact")
    elif meta.get("weight_schema_variant") == "lpbq_blockwise_expansion":
        raise ValueError(f"{case_dir}: LPBQ case requires --w4-encoding lpbq")

    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "runtime_inputs_native"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    activation_q = arr["activation_q"]
    if activation_q.shape != (m, k):
        raise ValueError(f"activation shape mismatch: {activation_q.shape} vs {(m, k)}")
    act_conv_q = activation_q.T.reshape(1, k, 1, m)
    act_dtype = np.dtype("<u2") if qparams["activation"]["bitwidth"] == 16 else np.dtype("uint8")
    act_conv_q.astype(act_dtype, copy=False).tofile(runtime_dir / "A.raw")

    activation_float = arr["activation_float"]
    if activation_float.shape != (m, k):
        raise ValueError(f"activation float shape mismatch: {activation_float.shape} vs {(m, k)}")
    activation_float.T.reshape(1, k, 1, m).astype("<f4").tofile(out_dir / "input_A.raw")

    weight_float = arr["weight_float_nk"]
    if weight_float.shape != (n, k):
        raise ValueError(f"weight float shape mismatch: {weight_float.shape} vs {(n, k)}")
    w_conv = weight_float.reshape(n, k, 1, 1).astype(np.float32)
    bias = arr["bias_float"].astype(np.float32)

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, k, 1, m])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n, 1, m])
    initializers = [
        numpy_helper.from_array(w_conv, name="W"),
        numpy_helper.from_array(bias, name="B"),
    ]
    conv = helper.make_node(
        "Conv",
        ["A", "W", "B"],
        ["Y"],
        name="conv1x1",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([conv], f"native_python_case_{family}", [a], [y], initializers)
    model = helper.make_model(
        graph,
        producer_name=f"qnn_native_python_case_{family}_{m}x{k}x{n}",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_dir / "case.onnx")

    act = qparams["activation"]
    out = qparams["output"]
    if w4_encoding == "lpbq":
        weight_encoding = lpbq_weight_enc_v1(
            "W",
            arr["weight_scale"],
            arr["weight_lpbq_per_block_int_scale"],
            int(qparams["weight"].get("block_size", 32)),
        )
    else:
        weight_encoding = weight_enc_v1("W", qparams["weight"]["bitwidth"], arr["weight_scale"])
    overrides = {
        "version": "1.0.0",
        "activation_encodings": [
            enc_v1("A", act["bitwidth"], act["scale"], act["zero_point"], symmetric=act["bitwidth"] == 16),
            enc_v1("Y", out["bitwidth"], out["scale"], out["zero_point"], symmetric=out["bitwidth"] == 16),
        ],
        "param_encodings": [
            weight_encoding,
        ],
    }
    (out_dir / "quant_overrides.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    (out_dir / "runtime_input_list.txt").write_text("A:=runtime_inputs_native/A.raw\n", encoding="utf-8")
    (out_dir / "input_list.txt").write_text("A:=input_A.raw\n", encoding="utf-8")

    native_io = {
        "case_dir": str(case_dir),
        "family": family,
        "kernel_family": meta.get("kernel_family"),
        "weight_schema_variant": meta.get("weight_schema_variant"),
        "input_name": "A",
        "output_name": "Y",
        "native_input": "runtime_inputs_native/A.raw",
        "runtime_input_list": "runtime_input_list.txt",
        "float_input": "input_A.raw",
        "native_input_storage": "uint16_le" if act["bitwidth"] == 16 else "uint8",
        "expected_native_output_storage": "uint16_le" if out["bitwidth"] == 16 else "uint8",
        "conv_input_shape": [1, k, 1, m],
        "conv_output_shape": [1, n, 1, m],
        "native_output_to_logical_mn": "Y.raw is Conv output [1,N,1,M]; compare as raw.reshape(N,M).T",
        "logical_matmul_shape_mkn": [m, k, n],
        "source_model_contract": "float Conv(A,W,B); quantization overrides encode A/Y/W; bias is a float op parameter quantized by qairt-quantizer",
        "w4_encoding": w4_encoding,
    }
    (out_dir / "native_io.json").write_text(json.dumps(native_io, indent=2), encoding="utf-8")
    return meta


def write_htp_config(out_dir: Path, qnn: Path, arch: str, soc_id: int) -> None:
    (out_dir / "htp_config.json").write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": str(qnn / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"),
                    "config_file_path": str(out_dir / "htp_backend_ext.json"),
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "htp_config_device.json").write_text(
        json.dumps(
            {
                "backend_extensions": {
                    "shared_library_path": "../libQnnHtpNetRunExtensions.so",
                    "config_file_path": "htp_backend_ext.json",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "htp_backend_ext.json").write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "dsp_arch": arch,
                        "soc_id": soc_id,
                        "pd_session": "unsigned",
                        "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def qnn_env(qnn: Path) -> dict[str, str]:
    env = os.environ.copy()
    py = str(qnn / "lib/python")
    lib = str(qnn / "lib/x86_64-linux-clang")
    bin_dir = str(qnn / "bin/x86_64-linux-clang")
    env["PYTHONPATH"] = py + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PATH"] = bin_dir + ":" + env["PATH"]
    return env


def extract_dlc_bias(dlc_path: Path, qnn: Path) -> np.ndarray:
    """Read the quantized Conv bias tensor emitted by qairt-quantizer."""
    qnn_python = str(qnn / "lib/python")
    if qnn_python not in sys.path:
        sys.path.insert(0, qnn_python)
    from qti.aisw.dlc_utils import snpe_dlc_utils

    reader = snpe_dlc_utils.modeltools.IrDlcReader()
    reader.open(str(dlc_path))
    try:
        for graph_name in reader.get_ir_graph_names():
            graph = reader.get_ir_graph(graph_name)
            for op in graph.get_ops():
                for tensor in op.inputs():
                    if tensor.name() == "B":
                        return np.array(tensor.get_data(), dtype=np.int32).reshape(-1)
    finally:
        reader.close()
    raise ValueError(f"{dlc_path}: missing quantized Conv B tensor")


def validate_native_lpbq_graph(out_dir: Path) -> dict[str, Any]:
    mapping = next((out_dir / "ctx").glob("*bottom_mapping.json"), None)
    before = next((out_dir / "ctx").glob("*bottom_mapping_graph_before.json"), None)
    if mapping is None or before is None:
        raise ValueError(f"{out_dir}: missing backend op mapping JSON for LPBQ validation")

    def node_types(path: Path) -> set[str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = data.get("graph", {}).get("nodes", {})
        return {
            str(node.get("type"))
            for node in nodes.values()
            if isinstance(node, dict) and node.get("type") is not None
        }

    before_types = node_types(before)
    final_types = node_types(mapping)
    required_before = "Conv2d_w_blk_exp_scale"
    required_final = "q::ConvLayer.opt.expand_block_quant_to_pc_int8_weights"
    if required_before not in before_types:
        raise ValueError(f"{out_dir}: native LPBQ graph-before missing {required_before}")
    if required_final not in final_types:
        raise ValueError(f"{out_dir}: native LPBQ graph missing {required_final}")
    summary = {
        "lpbq_native_graph": True,
        "graph_before_contains": required_before,
        "final_graph_contains": required_final,
        "mapping": str(mapping),
        "graph_before_mapping": str(before),
    }
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "native_lpbq_graph_check.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def compare(case_dir: Path, out_dir: Path, out_bits: int, maxdiff_tolerance: int | None) -> dict[str, Any]:
    case = load_case(case_dir)
    meta = case["meta"]
    ref = case["arrays"]["output_ref_q"]
    dtype = np.dtype("<u2") if out_bits == 16 else np.dtype("uint8")
    native_conv = np.fromfile(out_dir / "device_out" / "Y.raw", dtype=dtype)
    m, k, n = meta["shape_mkn"]
    if native_conv.size != m * n:
        raise ValueError(f"native output size mismatch: got {native_conv.size}, expected {m*n}")
    # qnn-net-run native output preserves the Conv output contract [1,N,1,M].
    # Logical MatMul output is [M,N], so transpose N-major raw storage.
    native = native_conv.reshape(1, n, 1, m).reshape(n, m).T.copy()
    ref = ref.reshape(m, n)
    diff = np.abs(native.astype(np.int64) - ref.astype(np.int64))
    maxdiff = int(diff.max()) if diff.size else 0
    out_qparams = meta["qparams"]["output"]
    out_scale = float(out_qparams["scale"])
    out_offset = int(out_qparams.get("qnn_offset", -int(out_qparams["zero_point"])))
    native_dequant = (native.astype(np.float64) + out_offset) * out_scale
    ref_dequant = (ref.astype(np.float64) + out_offset) * out_scale
    dequant_diff = np.abs(native_dequant - ref_dequant)
    max_dequant_diff = float(dequant_diff.max()) if dequant_diff.size else 0.0
    mean_dequant_diff = float(dequant_diff.mean()) if dequant_diff.size else 0.0
    accepted_maxdiff = accepted_maxdiff_for_case(meta, maxdiff_tolerance)
    accepted = maxdiff <= accepted_maxdiff
    summary = {
        "family": meta["family"],
        "case": meta["case"],
        "case_dir": str(case_dir),
        "out_dir": str(out_dir),
        "exact": int((native == ref).sum()),
        "total": int(ref.size),
        "maxdiff": maxdiff,
        "mean_absdiff": float(diff.mean()) if diff.size else 0.0,
        "int_exact_elements": int((native == ref).sum()),
        "int_total_elements": int(ref.size),
        "int_max_abs_delta": maxdiff,
        "int_mean_abs_delta": float(diff.mean()) if diff.size else 0.0,
        "dequant_float_max_abs_delta": max_dequant_diff,
        "dequant_float_mean_abs_delta": mean_dequant_diff,
        "accepted": accepted,
        "accepted_maxdiff": accepted_maxdiff,
        "assertion": "maxdiff <= accepted_maxdiff",
        "comparison_scope": "qnn_native_htp_vs_python_aimet_oracle_only",
        "same_hardware_policy": "custom_vs_native and handwritten_vs_custom must remain exact-output gates",
        "quantization_reference": meta.get("quantization_reference"),
        "dequant_float_reference": "final output dequantization: (q + qnn_offset) * output_scale",
        "output_scale": out_scale,
        "output_qnn_offset": out_offset,
        "native_sha256": meta["files"]["output_ref_q"]["raw_sha256"],
        "native_output_to_logical_mn": "Y.raw [1,N,1,M] -> raw.reshape(N,M).T",
    }
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    np.save(analysis_dir / "native_output_mn.npy", native)
    (analysis_dir / "python_native_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qnn-sdk-root", type=Path, default=Path(os.environ.get("QNN_SDK_ROOT", "tools/qnn-sdk")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "oneplus"))
    parser.add_argument("--arch", default=os.environ.get("ARCH", "v75"))
    parser.add_argument("--soc-id", type=int, default=int(os.environ.get("SOC_ID", "57")))
    parser.add_argument("--num-inferences", type=int, default=1)
    parser.add_argument("--w4-encoding", choices=("symmetric", "lpbq"), default="symmetric")
    parser.add_argument(
        "--maxdiff-tolerance",
        type=int,
        default=None,
        help="Override accepted integer-output maxdiff. Default uses the family/case policy.",
    )
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    meta = write_artifact(case_dir, out_dir, args.w4_encoding)
    qnn = args.qnn_sdk_root.resolve()
    env = qnn_env(qnn)
    write_htp_config(out_dir, qnn, args.arch, args.soc_id)

    converter = qnn / "bin/x86_64-linux-clang/qairt-converter"
    quantizer = qnn / "bin/x86_64-linux-clang/qairt-quantizer"
    ctxgen = qnn / "bin/x86_64-linux-clang/qnn-context-binary-generator"
    run(
        [
            str(converter),
            "-i",
            str(out_dir / "case.onnx"),
            "--target_backend",
            "HTP",
            "--enable_framework_trace",
            "--quantization_overrides",
            str(out_dir / "quant_overrides.json"),
            "--source_model_input_layout",
            "A",
            "NONTRIVIAL",
            "--desired_input_layout",
            "A",
            "NONTRIVIAL",
            "--source_model_output_layout",
            "Y",
            "NONTRIVIAL",
            "--desired_output_layout",
            "Y",
            "NONTRIVIAL",
            "-o",
            str(out_dir / "case_encoded.dlc"),
        ],
        env=env,
        log=out_dir / "_convert.log",
    )
    act_bits = str(meta["qparams"]["activation"]["bitwidth"])
    weight_bits = str(meta["qparams"]["weight"]["bitwidth"])
    run(
        [
            str(quantizer),
            "--input_dlc",
            str(out_dir / "case_encoded.dlc"),
            "--output_dlc",
            str(out_dir / "case.dlc"),
            "--enable_float_fallback",
            "--act_bitwidth",
            act_bits,
            "--weights_bitwidth",
            weight_bits,
            "--bias_bitwidth",
            "32",
            "--param_quantizer_schema",
            "symmetric",
            "--target_backend",
            "HTP",
            *(["--restrict_quantization_steps=-0x8000 0x7F7F"] if act_bits == "16" else []),
            *(["--pack_4_bit_weights"] if weight_bits == "4" or args.w4_encoding == "lpbq" else []),
        ],
        env=env,
        log=out_dir / "_quantize.log",
    )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    dlc_bias = extract_dlc_bias(out_dir / "case.dlc", qnn)
    np.save(analysis_dir / "dlc_bias_q_int32.npy", dlc_bias)
    dlc_bias.astype(np.int32).tofile(analysis_dir / "dlc_bias_q_int32.raw")
    ctx_dir = out_dir / "ctx"
    ctx_dir.mkdir(exist_ok=True)
    run(
        [
            str(ctxgen),
            "--backend",
            str(qnn / "lib/x86_64-linux-clang/libQnnHtp.so"),
            "--dlc_path",
            str(out_dir / "case.dlc"),
            "--binary_file",
            "case_native_ctx",
            "--output_dir",
            str(ctx_dir),
            "--config_file",
            str(out_dir / "htp_config.json"),
            "--profiling_level",
            "detailed",
            "--profiling_option",
            "optrace",
            "--save_backend_op_mapping",
        ],
        cwd=out_dir,
        env=env,
        log=out_dir / "_ctxgen.log",
    )
    if args.w4_encoding == "lpbq":
        validate_native_lpbq_graph(out_dir)
    if args.no_device:
        print(f"prepared {out_dir}")
        return 0

    remote = f"qnn_run/python_case_native_{meta['family']}_{meta['case']}"
    run(["ssh", args.device, f"rm -rf {remote} && mkdir -p {remote}/runtime_inputs_native"])
    ssh_write(args.device, f"{remote}/case_native_ctx.bin", ctx_dir / "case_native_ctx.bin")
    ssh_write(args.device, f"{remote}/htp_config.json", out_dir / "htp_config_device.json")
    ssh_write(args.device, f"{remote}/htp_backend_ext.json", out_dir / "htp_backend_ext.json")
    ssh_write(args.device, f"{remote}/input_list.txt", out_dir / "runtime_input_list.txt")
    ssh_write(args.device, f"{remote}/runtime_inputs_native/A.raw", out_dir / "runtime_inputs_native/A.raw")
    run(
        [
            "ssh",
            args.device,
            (
                f"cd {remote} && rm -rf out && "
                "LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ "
                "../qnn-net-run --backend ../libQnnHtp.so "
                "--retrieve_context case_native_ctx.bin --input_list input_list.txt "
                "--profiling_level detailed --profiling_option optrace --output_dir out "
                "--config_file htp_config.json --use_native_input_files --use_native_output_files "
                f"--num_inferences {args.num_inferences} --perf_profile burst 2>&1"
            ),
        ],
        log=out_dir / "_run.log",
    )
    ssh_read(args.device, f"{remote}/out/qnn-profiling-data_0.log", out_dir / "device_out/qnn-profiling-data_0.log")
    try:
        ssh_read(args.device, f"{remote}/out/Result_0/Y_native.raw", out_dir / "device_out/Y.raw")
    except subprocess.CalledProcessError:
        ssh_read(args.device, f"{remote}/out/Result_0/Y.raw", out_dir / "device_out/Y.raw")
    summary = compare(case_dir, out_dir, meta["qparams"]["output"]["bitwidth"], args.maxdiff_tolerance)
    print(
        f"{summary['family']}:{summary['case']} "
        f"int_exact={summary['int_exact_elements']}/{summary['int_total_elements']} "
        f"int_max_abs_delta={summary['int_max_abs_delta']} "
        f"accepted_max_int_delta={summary['accepted_maxdiff']} "
        f"accepted={summary['accepted']} "
        f"int_mean_abs_delta={summary['int_mean_abs_delta']:.6f} "
        f"dequant_float_max_abs_delta={summary['dequant_float_max_abs_delta']:.9g} "
        f"dequant_float_mean_abs_delta={summary['dequant_float_mean_abs_delta']:.9g}"
    )
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
