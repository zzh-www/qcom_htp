#!/usr/bin/env python3
"""Summarize W4A8 BNB raw cvt/writeback packet groups.

The W4A8 inline-asm replica intentionally keeps several mixed HMX/control
packets as raw .word directives.  This helper extracts the raw groups around
the accumulator conversion/writeback path so the dataflow document can cite a
stable packet inventory without pretending that every HMX word is decoded.
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


WORD_RE = re.compile(r"\.word\s+([^\\]+)\\n")
ADDR_RE = re.compile(r"@0x([0-9a-fA-F]+)")


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
        "conclusion": [
            "W4A8 BNB keeps cvt/writeback mixed HMX/control packets raw.",
            "The raw groups consistently sit between bias/control loads and mxmem(...):cm = cvt stores.",
            "The raw a6..dc/dd words do not match plain cvt.ub = acc(rX) encodings assembled with clang-19.",
            "The stable output evidence is U8 drain semantics; exact cvt mnemonic remains intentionally undeclared.",
        ],
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
