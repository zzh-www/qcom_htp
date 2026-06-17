#!/usr/bin/env python3
"""DEPRECATED (P2.1, 2026-06-17) — merged into the canonical scripts/htp_timeline.py.

pure-HMX (GDNSolveHMX) single-impl Perfetto SVG. Use:
  scripts/htp_timeline.py single purehmx <trace.raw> <out.svg>

The §5 stage map (and the MM-consumer-only / stage-5-PACK-leaf hard rules) now live in the single
source htp_timeline.py. This wrapper forwards to it.
Reproduce the trace: EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_TRACE" build, run, pull the T trace.raw.
"""
import sys, os
sys.argv = [sys.argv[0], "single", "purehmx"] + sys.argv[1:]
sys.stderr.write("[DEPRECATED] gdn_pure_perfetto_timeline.py -> scripts/htp_timeline.py single purehmx\n")
exec(open(os.path.join(os.path.dirname(__file__), "htp_timeline.py")).read())
