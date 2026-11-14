#!/bin/bash
# diag (estimator validation, train labels used ONLY as a reference) -> feat_v22 cache
set -uo pipefail
[ -e "$IPX/diag/ipx_diag_test.npz" ] || NPROC=${NPROC:-280} OUT=$IPX/diag python ipx_diag.py || echo DIAG_FAILED
OUT=$SCR/stage11_ch1/feat_v22 WORKERS=${WORKERS:-280} python prep_v22.py || { echo PREP_FAILED; exit 1; }
