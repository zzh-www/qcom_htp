#!/usr/bin/env bash
#
# Shared helpers for QAIRT two-stage quantization flows.
# Calibration path: qairt-converter -> float DLC, then qairt-quantizer
# --input_list -> quantized DLC.
# Encoding-driven path: qairt-converter --quantization_overrides -> encoded
# DLC, then qairt-quantizer --enable_float_fallback -> final DLC.

qairt_make_abs_input_list() {
    local src="$1"
    local dst="$2"
    python3 - "$src" "$dst" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
root = src.parent.resolve()
lines = []
for raw in src.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        lines.append(raw)
        continue
    parts = []
    for item in line.split():
        if ":=" not in item:
            path = Path(item)
            parts.append(str(path if path.is_absolute() else (root / path).resolve()))
            continue
        name, path_text = item.split(":=", 1)
        path = Path(path_text.strip())
        if not path.is_absolute():
            path = (root / path).resolve()
        parts.append(f"{name.strip()}:={path}")
    lines.append(" ".join(parts))
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

qairt_quantize_dlc() {
    local input_dlc="$1"
    local output_dlc="$2"
    local input_list="$3"
    local act_bits="$4"
    local weight_bits="$5"
    local bias_bits="$6"
    local pack_4bit="$7"
    local log_path="$8"
    shift 8
    local quantizer="${QAIRT_QUANTIZER:-qairt-quantizer}"
    if [ -n "${QNN_SDK_ROOT:-}" ] && [ -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-quantizer" ]; then
        quantizer="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-quantizer"
    fi

    local -a args=(
        --input_dlc "$input_dlc"
        --output_dlc "$output_dlc"
        --input_list "$input_list"
        --act_bitwidth "$act_bits"
        --weights_bitwidth "$weight_bits"
        --bias_bitwidth "$bias_bits"
        --param_quantizer_schema symmetric
        --target_backend HTP
    )
    if [ "$act_bits" = "16" ]; then
        args+=("--restrict_quantization_steps=-0x8000 0x7F7F")
    fi
    if [ "$pack_4bit" = "1" ]; then
        args+=(--pack_4_bit_weights)
    fi

    "$quantizer" "${args[@]}" "$@" > "$log_path" 2>&1
}

# Use this for DLCs that already carry complete encodings from
# qairt-converter --quantization_overrides.  It intentionally omits
# --input_list and --op_package_lib so qairt-quantizer does not run custom ops
# through the CPU backend while applying fallback/save/packing metadata.
qairt_quantize_encoded_dlc() {
    local input_dlc="$1"
    local output_dlc="$2"
    local act_bits="$3"
    local weight_bits="$4"
    local bias_bits="$5"
    local pack_4bit="$6"
    local log_path="$7"
    shift 7
    local quantizer="${QAIRT_QUANTIZER:-qairt-quantizer}"
    if [ -n "${QNN_SDK_ROOT:-}" ] && [ -x "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-quantizer" ]; then
        quantizer="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-quantizer"
    fi

    local -a args=(
        --input_dlc "$input_dlc"
        --output_dlc "$output_dlc"
        --enable_float_fallback
        --act_bitwidth "$act_bits"
        --weights_bitwidth "$weight_bits"
        --bias_bitwidth "$bias_bits"
        --param_quantizer_schema symmetric
        --target_backend HTP
    )
    if [ "$act_bits" = "16" ]; then
        args+=("--restrict_quantization_steps=-0x8000 0x7F7F")
    fi
    if [ "$pack_4bit" = "1" ]; then
        args+=(--pack_4_bit_weights)
    fi

    "$quantizer" "${args[@]}" "$@" > "$log_path" 2>&1
}
