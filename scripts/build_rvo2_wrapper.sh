#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVO2_ROOT="${RVO2_ROOT:-$ROOT/external/RVO2}"

if [[ ! -d "$RVO2_ROOT/src" ]]; then
  echo "RVO2 source tree not found at $RVO2_ROOT" >&2
  echo "Set RVO2_ROOT=/path/to/RVO2 or copy the RVO2 source tree into external/RVO2." >&2
  exit 1
fi

g++ -std=c++17 -O3 -fPIC -shared \
  -I"$RVO2_ROOT/src" \
  "$ROOT/external/rvo2_c_api.cpp" \
  "$RVO2_ROOT/src/Agent.cc" \
  "$RVO2_ROOT/src/KdTree.cc" \
  "$RVO2_ROOT/src/Line.cc" \
  "$RVO2_ROOT/src/Obstacle.cc" \
  "$RVO2_ROOT/src/RVOSimulator.cc" \
  "$RVO2_ROOT/src/Vector2.cc" \
  -o "$ROOT/external/librvo2_c_api.so"

echo "$ROOT/external/librvo2_c_api.so"
