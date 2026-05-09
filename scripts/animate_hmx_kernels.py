#!/usr/bin/env python3
"""ASCII matrix animations for the HMX mixed-precision dataflow documentation.

The animations use small toy matrices to make the data movement visible.  They
are not an HMX emulator and are not intended to reproduce native packet timing.

Examples:
  python3 scripts/animate_hmx_kernels.py --kernel all
  python3 scripts/animate_hmx_kernels.py --kernel w4a8 --delay 0.15
  python3 scripts/animate_hmx_kernels.py --kernel w4a16 --no-clear --delay 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable


Matrix = list[list[int]]
CellMatrix = list[list[object]]


@dataclass(frozen=True)
class Frame:
    kernel: str
    stage: str
    lines: list[str]


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def signed4(value: int) -> int:
    value &= 0xF
    return value - 16 if value & 0x8 else value


def nibble(value: int) -> int:
    return value & 0xF


def pack_i4_lohi(lo: int, hi: int) -> int:
    return nibble(lo) | (nibble(hi) << 4)


def pack_cell(lo: int, hi: int) -> str:
    return f"{lo:+d}/{hi:+d}:{pack_i4_lohi(lo, hi):02X}"


def carrier_cell(native_byte: int) -> str:
    return f"{native_byte ^ 0x80:02X}"


def zeros(rows: int, cols: int) -> Matrix:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def matmul_frames(a: Matrix, w: Matrix, bias: list[int] | None = None) -> list[Matrix]:
    rows = len(a)
    k_total = len(a[0])
    cols = len(w[0])
    bias = bias or [0] * cols
    acc = zeros(rows, cols)
    frames = []
    for k in range(k_total):
        for m in range(rows):
            for n in range(cols):
                acc[m][n] += a[m][k] * w[k][n]
        frames.append([row[:] for row in acc])
    frames.append([[acc[m][n] + bias[n] for n in range(cols)] for m in range(rows)])
    return frames


def labels(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(count)]


def render_matrix(
    name: str,
    matrix: CellMatrix,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    highlights: set[tuple[int, int]] | None = None,
) -> list[str]:
    highlights = highlights or set()
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    row_labels = row_labels or [str(i) for i in range(rows)]
    col_labels = col_labels or [str(i) for i in range(cols)]

    rendered: list[list[str]] = []
    for r, row in enumerate(matrix):
        rendered_row = []
        for c, value in enumerate(row):
            text = str(value)
            if (r, c) in highlights:
                text = f"[{text}]"
            rendered_row.append(text)
        rendered.append(rendered_row)

    row_header = max(len(x) for x in [""] + row_labels)
    widths = []
    for c in range(cols):
        width = len(col_labels[c])
        for r in range(rows):
            width = max(width, len(rendered[r][c]))
        widths.append(width)

    def border() -> str:
        parts = ["-" * (row_header + 2)]
        parts.extend("-" * (w + 2) for w in widths)
        return "+" + "+".join(parts) + "+"

    out = [name, border()]
    header = [" " * row_header]
    header.extend(col_labels[c].center(widths[c]) for c in range(cols))
    out.append("| " + " | ".join(header) + " |")
    out.append(border())
    for r in range(rows):
        cells = [row_labels[r].rjust(row_header)]
        cells.extend(rendered[r][c].rjust(widths[c]) for c in range(cols))
        out.append("| " + " | ".join(cells) + " |")
    out.append(border())
    return out


def render_ports(body_family: str, act: str, weight: str, drain: str) -> list[str]:
    return [
        f"body family : {body_family}",
        f"activation  : {act}",
        f"weight      : {weight}",
        "MAC fabric  : internal accumulator banks",
        f"drain       : {drain}",
        "",
        "activation port ----\\",
        "                    +--> MAC fabric --> accumulator banks --> drain --> output",
        "weight port     ----/",
    ]


def frame(kernel: str, stage: str, *blocks: Iterable[str]) -> Frame:
    lines: list[str] = []
    for block in blocks:
        lines.extend(block)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return Frame(kernel=kernel, stage=stage, lines=lines)


def render_frame(f: Frame, index: int, total: int) -> str:
    header = f"=== [{index + 1:02d}/{total:02d}] {f.kernel}: {f.stage} ==="
    return "\n".join([header, *f.lines])


def u8i8_frames() -> list[Frame]:
    kernel = "u8i8 / convbbb / byte-byte-byte"
    a = [
        [2, 5, 1, 3],
        [4, 1, 0, 2],
    ]
    w = [
        [3, -2, 1],
        [-1, 4, 2],
        [2, 0, -3],
        [1, -1, 5],
    ]
    bias = [1, 2, 0]
    accum = matmul_frames(a, w, bias)

    frames = [
        frame(
            kernel,
            "typed HMX route",
            render_ports(
                "convbbb",
                "U8 activation tile stream",
                "I8 byte-packed weight stream",
                "convert accumulator to U8",
            ),
        ),
        frame(
            kernel,
            "logical matrices",
            render_matrix("A[M,K] as U8", a, labels("m", 2), labels("k", 4)),
            render_matrix("W[K,N] as I8 bytes", w, labels("k", 4), labels("n", 3)),
            ["bias/control for U8 drain: " + str(bias)],
        ),
        frame(
            kernel,
            "weight stream stays byte-granular",
            render_matrix("physical W stream: one signed byte per weight", w, labels("k", 4), labels("n", 3)),
            ["No nibble unpack path is needed here: the weight port reads byte lanes."],
        ),
    ]

    for k, acc in enumerate(accum[:-1]):
        frames.append(
            frame(
                kernel,
                f"K reduce step k{k}",
                render_matrix("A column entering activation port", a, labels("m", 2), labels("k", 4), {(0, k), (1, k)}),
                render_matrix("W row entering weight port", w, labels("k", 4), labels("n", 3), {(k, 0), (k, 1), (k, 2)}),
                render_matrix("accumulator banks after this step", acc, labels("m", 2), labels("n", 3)),
                [f"operation: acc[m,n] += A[m,k{k}] * W[k{k},n]"],
            )
        )

    drained = [[clamp(x, 0, 255) for x in row] for row in accum[-1]]
    frames.append(
        frame(
            kernel,
            "drain to U8 output",
            render_matrix("accumulator + bias", accum[-1], labels("m", 2), labels("n", 3)),
            render_matrix("C[M,N] as U8", drained, labels("m", 2), labels("n", 3)),
            ["drain recipe: add folded bias/control, round, saturate to [0,255], store U8"],
        )
    )
    return frames


def w4a8_frames() -> list[Frame]:
    kernel = "w4a8 / convbnb / byte-nibble-byte"
    a = [
        [7, 2, 5, 1],
        [3, 6, 2, 4],
    ]
    w = [
        [1, -2, 3, 0, -1, 2, -3, 1],
        [-4, 1, 0, 2, 3, -2, 1, -1],
        [2, 0, -1, 3, -2, 1, 2, -4],
        [1, -3, 2, -2, 0, 3, -1, 2],
    ]
    bias = [4, 2, 0, 1, 3, 2, 1, 0]
    accum = matmul_frames(a, w, bias)

    packed = [
        [pack_cell(w[k][n], w[k][n + 4]) for n in range(4)]
        for k in range(4)
    ]
    carrier = [
        [carrier_cell(pack_i4_lohi(w[k][n], w[k][n + 4])) for n in range(4)]
        for k in range(4)
    ]

    frames = [
        frame(
            kernel,
            "typed HMX route",
            render_ports(
                "convbnb",
                "U8 Crouton_8 activation blocks",
                "I4 nibbles packed in a byte carrier",
                "convert accumulator to U8",
            ),
        ),
        frame(
            kernel,
            "logical matrices",
            render_matrix("A[M,K] as U8", a, labels("m", 2), labels("k", 4)),
            render_matrix("W[K,N] as signed I4", w, labels("k", 4), labels("n", 8)),
            ["toy uses N=8, so real n+32 pairing is shown as n+4 pairing."],
        ),
        frame(
            kernel,
            "pack W4A8 as output-channel pairs",
            render_matrix("native W4 bytes: pack(W[k,n], W[k,n+4])", packed, labels("k", 4), ["n0/n4", "n1/n5", "n2/n6", "n3/n7"]),
            [
                "real kernel scale: K32 x N64 tile, one byte pairs n and n+32.",
                "this is the nibble stream consumed by the convbnb weight port.",
            ],
        ),
        frame(
            kernel,
            "QNN carrier sidecar view",
            render_matrix("DLC carrier byte = native byte ^ 0x80", carrier, labels("k", 4), ["n0/n4", "n1/n5", "n2/n6", "n3/n7"]),
            [
                "weights_to_vtcm toggles the sign bit again.",
                "HMX finally sees the native packed nibble bytes from the previous frame.",
            ],
        ),
    ]

    for k, acc in enumerate(accum[:-1]):
        frames.append(
            frame(
                kernel,
                f"K reduce step k{k}",
                render_matrix("A column entering activation port", a, labels("m", 2), labels("k", 4), {(0, k), (1, k)}),
                render_matrix("logical W row decoded from nibble stream", w, labels("k", 4), labels("n", 8), {(k, c) for c in range(8)}),
                render_matrix("accumulator banks after this step", acc, labels("m", 2), labels("n", 8)),
                [f"operation: acc[m,n] += A[m,k{k}] * I4(W[k{k},n])"],
            )
        )

    drained = [[clamp(x, 0, 255) for x in row] for row in accum[-1]]
    frames.append(
        frame(
            kernel,
            "drain to U8 output",
            render_matrix("accumulator + bias", accum[-1], labels("m", 2), labels("n", 8)),
            render_matrix("C[M,N] as U8", drained, labels("m", 2), labels("n", 8)),
            ["drain is byte output; W4 only changed the weight-side route."],
        )
    )
    return frames


def w4a16_frames() -> list[Frame]:
    kernel = "w4a16 / convhnh / half-nibble-half"
    a = [
        [12, 7, 3, 9, 5, 2, 6, 4],
        [4, 11, 8, 1, 7, 3, 2, 10],
    ]
    w = [
        [1, -2, 0, 3],
        [-1, 2, -3, 1],
        [2, 0, 1, -2],
        [0, -1, 3, 2],
        [-2, 1, 2, 0],
        [3, -3, 1, -1],
        [1, 2, -2, 3],
        [-1, 0, 2, -3],
    ]
    bias = [16, 8, 4, 0]
    accum = matmul_frames(a, w, bias)

    packed = [
        [pack_cell(w[k][n], w[k + 4][n]) for n in range(4)]
        for k in range(4)
    ]

    frames = [
        frame(
            kernel,
            "typed HMX route",
            render_ports(
                "convhnh",
                "U16 activation tile stream",
                "I4 nibbles packed in the HNH weight stream",
                "convert accumulator to U16",
            ),
        ),
        frame(
            kernel,
            "logical matrices",
            render_matrix("A[M,K] as U16", a, labels("m", 2), labels("k", 8)),
            render_matrix("W[K,N] as signed I4", w, labels("k", 8), labels("n", 4)),
            ["toy K=8 mirrors the draft's k and k+4 packing example."],
        ),
        frame(
            kernel,
            "pack W4A16 as K-pairs per output column",
            render_matrix("native-style HNH bytes: pack(W[k,n], W[k+4,n])", packed, ["k0/k4", "k1/k5", "k2/k6", "k3/k7"], labels("n", 4)),
            [
                "real kernel scale: K32 blocks with an HNH-specific nibble stream.",
                "activation and output are halfword-side streams, not byte output streams.",
            ],
        ),
    ]

    for k, acc in enumerate(accum[:-1]):
        frames.append(
            frame(
                kernel,
                f"K reduce step k{k}",
                render_matrix("A column entering activation port", a, labels("m", 2), labels("k", 8), {(0, k), (1, k)}),
                render_matrix("logical W row decoded from nibble stream", w, labels("k", 8), labels("n", 4), {(k, c) for c in range(4)}),
                render_matrix("accumulator banks after this step", acc, labels("m", 2), labels("n", 4)),
                [f"operation: acc[m,n] += A16[m,k{k}] * I4(W[k{k},n])"],
            )
        )

    drained = [[clamp(x, 0, 65535) for x in row] for row in accum[-1]]
    frames.append(
        frame(
            kernel,
            "drain to U16 output",
            render_matrix("accumulator + bias", accum[-1], labels("m", 2), labels("n", 4)),
            render_matrix("C[M,N] as U16", drained, labels("m", 2), labels("n", 4)),
            ["drain recipe: A16/W4 control converts internal accumulators to U16 output."],
        )
    )
    return frames


KERNEL_BUILDERS = {
    "u8i8": u8i8_frames,
    "w4a8": w4a8_frames,
    "w4a16": w4a16_frames,
}


def selected_frames(kernel: str) -> list[Frame]:
    if kernel == "all":
        frames: list[Frame] = []
        for name in ("u8i8", "w4a8", "w4a16"):
            frames.extend(KERNEL_BUILDERS[name]())
        return frames
    return KERNEL_BUILDERS[kernel]()


def play(frames: list[Frame], delay: float, clear: bool, max_frames: int | None) -> None:
    total = len(frames) if max_frames is None else min(len(frames), max_frames)
    for i, f in enumerate(frames[:total]):
        if clear:
            sys.stdout.write("\033[2J\033[H")
        elif i:
            sys.stdout.write("\n\n" + "=" * 80 + "\n\n")
        sys.stdout.write(render_frame(f, i, total) + "\n")
        sys.stdout.flush()
        if delay > 0 and i + 1 < total:
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ASCII matrix animations for HMX u8i8/w4a8/w4a16 dataflows."
    )
    parser.add_argument(
        "--kernel",
        choices=["all", *KERNEL_BUILDERS.keys()],
        default="all",
        help="which kernel dataflow to animate",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="seconds to wait between frames; use 0 for dump mode",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="print frames sequentially instead of clearing the terminal",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="debug/test helper: stop after this many frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clear = not args.no_clear and sys.stdout.isatty() and os.environ.get("TERM") not in ("", "dumb")
    frames = selected_frames(args.kernel)
    play(frames, delay=max(0.0, args.delay), clear=clear, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
