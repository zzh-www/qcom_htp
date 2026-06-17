#!/usr/bin/env python3
"""DEPRECATED (P2.1, 2026-06-17) — merged into the canonical scripts/htp_timeline.py.

ASCII per-thread timeline. Use:
  scripts/htp_timeline.py ascii <impl: ship|ares|hvxmix|purehmx> <trace.raw> [width]

The §5 stage map + MM-consumer-only rule now live in the single source htp_timeline.py.
Default impl assumed = purehmx (the production solve) when none is given, for back-compat.
"""
import sys, os
# back-compat: old call was `gdn_pipe_timeline.py T.raw [width]` (no impl arg, pure-HMX trace).
args = sys.argv[1:]
impl = "purehmx"
if args and args[0] in ("ship", "ares", "hvxmix", "purehmx"):
    impl = args.pop(0)
sys.argv = [sys.argv[0], "ascii", impl] + args
sys.stderr.write("[DEPRECATED] gdn_pipe_timeline.py -> scripts/htp_timeline.py ascii <impl> <trace.raw>\n")
exec(open(os.path.join(os.path.dirname(__file__), "htp_timeline.py")).read())
