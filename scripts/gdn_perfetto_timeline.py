#!/usr/bin/env python3
"""DEPRECATED (P2.1, 2026-06-17) — merged into the canonical scripts/htp_timeline.py.

The original GDNSolveHVXMixHMX single-swimlane Perfetto SVG. Use:
  scripts/htp_timeline.py single hvxmix <trace.raw> <out.svg>

This script's stage map and palette are superseded by the §5 single-source table in htp_timeline.py.
This wrapper forwards to it.
Reproduce the trace: build with -DGDN_BR_TRACE, run, pull the T trace.raw.
"""
import sys, os
sys.argv = [sys.argv[0], "single", "hvxmix"] + sys.argv[1:]
sys.stderr.write("[DEPRECATED] gdn_perfetto_timeline.py -> scripts/htp_timeline.py single hvxmix\n")
exec(open(os.path.join(os.path.dirname(__file__), "htp_timeline.py")).read())
