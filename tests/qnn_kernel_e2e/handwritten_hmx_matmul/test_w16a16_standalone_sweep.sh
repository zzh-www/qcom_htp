#!/usr/bin/env bash
# Multi-shape CI gate for the QNN-free W16A16 standalone (handwriting route).
#
# Runs scripts/w16a16_standalone_shape_sweep.sh over a representative subset:
# the 128-N split path, the single-call path (N%128!=0), and K-variation -- all
# at M=256 (the supported standalone envelope; M<256 needs per-shape Crouton
# padding, covered for the productized op by scripts/w16a16_shape_sweep.sh).
# Needs device (native refs via profile_all.sh). Full set via SHAPES env.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
SHAPES="${SHAPES:-256,256,256 256,256,96 256,64,128}" \
  DEVICE="${DEVICE:-oneplus}" \
  bash "$ROOT_DIR/scripts/w16a16_standalone_shape_sweep.sh"
