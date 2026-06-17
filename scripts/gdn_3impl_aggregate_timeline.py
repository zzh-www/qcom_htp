#!/usr/bin/env python3
"""DEPRECATED (P2.1, 2026-06-17) — merged into the canonical scripts/htp_timeline.py.

This was the SEED for the unified tool (the only one that already handled per-impl stage mapping).
Its logic now lives in htp_timeline.py `aggregate` mode, driven by the §5 single-source stage table.

Replacement (byte-identical timeline rects, verified):
  scripts/htp_timeline.py aggregate <ship.raw> <ares.raw> <purehmx.raw> <out.svg>

This wrapper forwards to that so existing call sites keep working.
"""
import sys, os
sys.argv = [sys.argv[0], "aggregate"] + sys.argv[1:]
sys.stderr.write("[DEPRECATED] gdn_3impl_aggregate_timeline.py -> scripts/htp_timeline.py aggregate\n")
exec(open(os.path.join(os.path.dirname(__file__), "htp_timeline.py")).read())
