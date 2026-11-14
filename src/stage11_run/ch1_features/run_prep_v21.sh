#!/bin/bash
# FEATURE-V2.1: contract tests -> full prep (train 6,891 / holdout 150 / test 874) -> predict-path verify.
set -uo pipefail
python test_contract.py || exit 1
python prep_v21.py || exit 1
# the inputs-only predict path (as it will run on the Final shots) must reproduce the cache bit-for-bit
python build_unseen.py --in-dir "$TEST_DIR" --out "$OUT/_unseen_check" --scaler "$OUT/scaler_v21.npz" \
    --limit 60 --workers 60 --verify-against "$OUT/test"
