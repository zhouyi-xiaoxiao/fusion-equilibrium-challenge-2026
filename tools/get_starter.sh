#!/bin/bash
# Fetch the organisers' starter kit at the pinned commit into src/starter (official scorer, validator, data_fixes.py).
# It is not vendored in this repository; the code here imports it unmodified.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
URL=https://github.com/Sophelio/fusion-equilibrium-challenge-starter
SHA=a67429165b09eb81c311d44db6ff11743f108b0e
D=$REPO/src/starter
if [ -d "$D/.git" ] && [ "$(git -C "$D" rev-parse HEAD)" = "$SHA" ]; then echo "starter at $SHA"; exit 0; fi
rm -rf "$D"; git clone --quiet "$URL" "$D"; git -C "$D" checkout --quiet "$SHA"
[ "$(git -C "$D" rev-parse HEAD)" = "$SHA" ] && echo "starter at $SHA"
