#!/usr/bin/env bash
# Value-distribution CI gate for the QNN-free W16A16 standalone (handwriting).
#
# Runs scripts/w16a16_standalone_dist_sweep.sh over a representative subset of
# weight/activation value distributions + seeds (uniform multi-seed, signs,
# zeros) and asserts byte-exact vs native. Ensures correctness is not specific
# to a single random draw. Needs device (native refs via profile_all.sh).
# Full set / extra dists via CONFIGS env. See the sweep script header for the
# edge cases extreme(+-max) / impulse that originally exposed the int16 weight
# high-byte clip bug (now fixed: q16 clipped to 32639).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIGS="${CONFIGS:-uniform:42 extreme:0 impulse:0}" \
  bash "$ROOT_DIR/scripts/w16a16_standalone_dist_sweep.sh"
