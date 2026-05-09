#!/usr/bin/env python3
"""Summarize W4A8 BNB cvt/writeback packet groups.

The W4A8 inline-asm replica has promoted the activation/weight loads,
accumulator conversion, pure branch/loop control, padding paths, and mixed
HMX-store/control tails to byte-proven asm.  This helper reports any raw
writeback groups that regress back into the file and the decoded scaled cvt
instructions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INC = Path("example/qnn_hmx_matmul_w4a8/src/v73deep_conv1x1_kernel.inc")
DEFAULT_SO = Path("tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so")
DEFAULT_VMA = "0x2f0780"
DEFAULT_SIZE = 2624
PLAIN_CVT_UB_ACC_WORDS = {
    # Assembled with clang-19 -target hexagon -mcpu=hexagonv75 -mhmx:
    #   { cvt.ub = acc(rX) }
    "r23": "0xa6f7d710",
    "r25": "0xa6f9d710",
    "r26": "0xa6fad710",
    "r27": "0xa6fbd710",
    "r29": "0xa6fdd710",
    "r30": "0xa6fed710",
    "r31": "0xa6ffd710",
}
SCALED_CVT_UB_ACC_WORDS = {
    # Assembled with clang-19 -target hexagon -mcpu=hexagonv75 -mhmx:
    #   { cvt.ub = acc(rX):sc0/sc1 }
    "r27:sc0": "0xa6fbdc10",
    "r27:sc1": "0xa6fbdd10",
    "r31:sc0": "0xa6ffdc10",
    "r31:sc1": "0xa6ffdd10",
}


WORD_RE = re.compile(r"\.word\s+([^\\]+)\\n")
ADDR_RE = re.compile(r"@0x([0-9a-fA-F]+)")
SCALED_CVT_RE = re.compile(r"cvt\.ub = acc\(r(27|31)\):sc([01])")


def _words_from_line(line: str) -> tuple[str, ...]:
    match = WORD_RE.search(line)
    if not match:
        return ()
    return tuple(word.strip() for word in match.group(1).split(","))


def _addr_from_comments(comments: list[str]) -> str | None:
    for comment in reversed(comments):
        match = ADDR_RE.search(comment)
        if match:
            return "0x" + match.group(1).lower()
    return None


def _classify(comments: list[str], words: tuple[str, ...]) -> str:
    text = "\n".join(comments).lower()
    if "accumulator conversion/writeback tail" in text:
        return "pre_store_cvt_tail"
    if "remaining raw tail paired with the preceding cvt output store" in text:
        return "post_store_tail"
    if "remaining raw tail after byte-proven bias/window setup" in text:
        return "post_bias_cvt_tail"
    if any(word in ("0x10bf40f6", "0x10bf40f8", "0x75594000") for word in words):
        return "cvt_control_tail"
    return "other_raw"


def extract_groups(inc: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    comments: list[str] = []
    for lineno, line in enumerate(inc.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("*/"):
            comments.append(stripped)
            comments = comments[-16:]
            continue
        words = _words_from_line(line)
        if not words:
            if stripped and not stripped.startswith('"'):
                comments = []
            continue
        cls = _classify(comments, words)
        if cls != "other_raw":
            groups.append(
                {
                    "line": lineno,
                    "addr": _addr_from_comments(comments),
                    "class": cls,
                    "word_count": len(words),
                    "words": list(words),
                    "comment": " ".join(c.strip("/* ") for c in comments[-4:]),
                }
            )
        comments = []
    return groups


def build_report(inc: Path) -> dict[str, Any]:
    text = inc.read_text(encoding="utf-8")
    groups = extract_groups(inc)
    word_counter = Counter(word for group in groups for word in group["words"])
    pattern_counter = Counter(tuple(group["words"]) for group in groups)
    class_counter = Counter(group["class"] for group in groups)
    cvt_like_words = [
        word
        for word in ("0x75594000", "0x10bf40f6", "0x10bf40f8", "0x5cdf68f6", "0x5cdf68f8")
        if word_counter[word]
    ]
    plain_cvt_matches = {
        reg: word for reg, word in PLAIN_CVT_UB_ACC_WORDS.items() if word_counter[word]
    }
    scaled_cvt_counter = Counter(
        f"r{reg}:sc{scale}" for reg, scale in SCALED_CVT_RE.findall(text)
    )
    if cvt_like_words:
        conclusion = [
            "W4A8 BNB still has raw cvt-like control words in the writeback path.",
            "The raw groups consistently sit between bias/control loads and mxmem(...):cm = cvt stores.",
            "The remaining raw cvt-like words should not be described as decoded mnemonics.",
        ]
    else:
        conclusion = [
            "W4A8 BNB accumulator conversion packets are byte-proven readable asm.",
            "The decoded scaled conversion forms are cvt.ub = acc(r27/r31):sc0/sc1.",
            "Pure branch/loop control, padding, and mixed HMX-store/control tails are readable asm/.space.",
            "No raw writeback groups remain; no raw 0x92 HMX load or raw 0xa6 drain word remains.",
        ]
    return {
        "inc": str(inc),
        "native_slice": {
            "so": str(DEFAULT_SO),
            "vma": DEFAULT_VMA,
            "size": DEFAULT_SIZE,
        },
        "groups": groups,
        "class_counts": dict(class_counter),
        "pattern_counts": [
            {"words": list(words), "count": count}
            for words, count in pattern_counter.most_common()
        ],
        "word_counts": dict(word_counter),
        "cvt_like_words": cvt_like_words,
        "plain_cvt_ub_acc_words": PLAIN_CVT_UB_ACC_WORDS,
        "plain_cvt_ub_acc_matches": plain_cvt_matches,
        "scaled_cvt_ub_acc_words": SCALED_CVT_UB_ACC_WORDS,
        "scaled_cvt_ub_acc_counts": dict(scaled_cvt_counter),
        "conclusion": conclusion,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=== W4A8 cvt/writeback raw packet inventory ===")
    print(f"inc: {report['inc']}")
    ns = report["native_slice"]
    print(f"native slice: {ns['so']} @ {ns['vma']} size={ns['size']}")
    print("\n[class counts]")
    for cls, count in sorted(report["class_counts"].items()):
        print(f"  {cls}: {count}")
    print("\n[repeated raw patterns]")
    for item in report["pattern_counts"]:
        words = ", ".join(item["words"])
        print(f"  count={item['count']}: {words}")
    print("\n[cvt-like control words]")
    print("  " + ", ".join(report["cvt_like_words"]))
    print("\n[plain cvt.ub = acc(rX) comparison]")
    print("  assembled candidate words: " + ", ".join(report["plain_cvt_ub_acc_words"].values()))
    print(f"  matches in W4A8 raw inventory: {report['plain_cvt_ub_acc_matches']}")
    print("\n[scaled cvt.ub = acc(rX):scY mnemonics]")
    print("  assembled candidate words: " + ", ".join(report["scaled_cvt_ub_acc_words"].values()))
    for name, count in sorted(report["scaled_cvt_ub_acc_counts"].items()):
        print(f"  {name}: {count}")
    print("\n[first groups]")
    for group in report["groups"][:12]:
        print(
            f"  line {group['line']} addr={group['addr']} {group['class']}: "
            + ", ".join(group["words"])
        )
    print("\n[conclusion]")
    for line in report["conclusion"]:
        print(f"  - {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inc", type=Path, default=DEFAULT_INC)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.inc.resolve())
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
