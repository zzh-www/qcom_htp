#!/usr/bin/env python3
"""DEPRECATED (P2.1, 2026-06-17) — merged into the canonical scripts/htp_timeline.py.

GDNSolveHVXMixHMX single-impl Perfetto SVG. Use:
  scripts/htp_timeline.py single hvxmix <trace.raw> <out.svg>   (or 'ship' / 'ares' for titled variants)

NOTE the口径 fix vs this old script: §5 makes stage 5 (PREP) a PACK LEAF (this old script put it in
CONTAINER, which掏空ed the HVXMix producer rows ~20-30pp and faked low producer occupancy). The
canonical tool draws stage 5 as a PACK leaf in BOTH taxonomies, so the routes render on the SAME口径.
This wrapper forwards to the canonical tool.
Reproduce the trace: see docs/cycle_metric_alignment.md §5 / the HVXMix build flags in perf_3impl.
"""
import sys, os
sys.argv = [sys.argv[0], "single", "hvxmix"] + sys.argv[1:]
sys.stderr.write("[DEPRECATED] gdn_hvxmix_perfetto_timeline.py -> scripts/htp_timeline.py single hvxmix\n")
exec(open(os.path.join(os.path.dirname(__file__), "htp_timeline.py")).read())
