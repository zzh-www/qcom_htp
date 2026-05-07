#!/usr/bin/env python3
"""Generate a native W4A16 Conv graph with a QHPI tensor dump after Conv."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper

DOMAIN = "hmx"
DUMP_OP = "HmxW4A16TensorDump"


def _ensure_custom_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == DOMAIN:
            return
    model.opset_import.append(helper.make_opsetid(DOMAIN, 1))


def _shape_from_value_info(value: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_param if dim.dim_param else dim.dim_value)
    return dims


def _append_u16_output(model: onnx.ModelProto, source: str, output_name: str) -> None:
    for value in list(model.graph.output) + list(model.graph.value_info) + list(model.graph.input):
        if value.name != source:
            continue
        model.graph.output.append(
            helper.make_tensor_value_info(output_name, TensorProto.UINT16, _shape_from_value_info(value))
        )
        return
    raise ValueError(f"missing value info for {source}")


def _copy_quant_overrides(src: Path, dst: Path, dump_output_name: str) -> None:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    activations = data.setdefault("activation_encodings", {})
    if "Y" in activations and dump_output_name not in activations:
        activations[dump_output_name] = activations["Y"]
    dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(
    native_dir: Path,
    out_dir: Path,
    dump_output_name: str = "D",
) -> None:
    native_onnx = native_dir / "conv.onnx"
    native_quant = native_dir / "quant_overrides.json"
    native_input = native_dir / "input_A.raw"
    native_input_list = native_dir / "input_list.txt"
    for path in (native_onnx, native_quant, native_input, native_input_list):
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(native_onnx)
    if not model.graph.node:
        raise ValueError(f"{native_onnx} contains no nodes")
    conv = model.graph.node[0]
    if conv.op_type != "Conv":
        raise ValueError(f"expected first node to be Conv, got {conv.op_type}")
    if not conv.output or conv.output[0] != "Y":
        raise ValueError(f"expected Conv output Y, got {list(conv.output)}")

    _append_u16_output(model, "Y", dump_output_name)
    model.graph.node.append(
        helper.make_node(
            DUMP_OP,
            inputs=["Y"],
            outputs=[dump_output_name],
            name="dump_native_conv_y",
            domain=DOMAIN,
        )
    )
    _ensure_custom_opset(model)
    onnx.save(model, out_dir / "native_conv_tensor_dump.onnx")

    _copy_quant_overrides(native_quant, out_dir / "quant_overrides.json", dump_output_name)
    shutil.copy2(native_input, out_dir / "input_A.raw")
    shutil.copy2(native_input_list, out_dir / "input_list.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=Path("../../../qnn_matmul_profile/output_codex_native_w4a16_same_custom_256"),
        help="Directory containing conv.onnx, quant_overrides.json, input_A.raw, and input_list.txt.",
    )
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--dump-output-name", default="D")
    args = parser.parse_args()

    native_dir = args.native_dir
    if not native_dir.is_absolute():
        native_dir = (Path.cwd() / native_dir).resolve()
    generate(native_dir, args.out_dir.resolve(), args.dump_output_name)


if __name__ == "__main__":
    main()
