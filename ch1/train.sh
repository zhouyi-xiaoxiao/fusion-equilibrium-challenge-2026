#!/bin/bash
# Ch1 training entrypoint: retrains every learned member the predict path of the selected profile loads, from scratch, on all 7,041
# labelled DIII-D shots (fixed epochs and seeds, no validation, no early stopping).
#
#   [CH1_PROFILE=<profile>] bash ch1/train.sh [MODEL_ROOT]          (default $SCR/stage11_ch1/final_path/train_<UTC stamp>)
#
#   base     FM 11 psi nets + TW 4 seeds + SC 17 members on the mirror-canonical FEATURE-V2.2 cache    -> MODEL_ROOT/base
#   heads    profile fccx:    canonical direct heads, residual MLP 768x4, 6 seeds x 42 epochs, FEATURE-V2.1 cache   -> MODEL_ROOT/heads
#            profile hc10x24: canonical direct heads, residual MLP 1024x8, dropout 0.15, 24 seeds (4 groups x 6) x 42 epochs, on the
#                             FEATURE-V2.2 cache built by ch1/build_feat22.py                                        -> MODEL_ROOT/heads22
#   OOF=1    additionally refits the out-of-fold teacher statistics (cov / FD steps / innovation scales)   -> MODEL_ROOT/oof
# Prerequisites = generated data (feature caches, metric files): MANIFESTS=weights/GENERATED.sha256 bash tools/fetch_weights.sh ...
# places and verifies them (docs/CH1_GENERATED_DATA.md); the PCA bases, scalers and frozen constants come with the weights.
# Local mode (default): runs on this machine with NGPU GPUs (default 1).  USE_SLURM=1: SLURM jobs as in
# src/stage11_run/ch1_package_final/train.sh (base 4 GPUs; heads 1 GPU, or 4 x 1 GPU for hc10x24).  CMP=1 compares the new weights
# with the manifest weights under $SCR.
# Then:  BASE_ROOT=MODEL_ROOT/base HEAD_ROOT=MODEL_ROOT/heads [HEAD22_ROOT=MODEL_ROOT/heads22] [OOF_ROOT=MODEL_ROOT/oof] bash ch1/predict.sh <shot_dir> <name>
set -euo pipefail
. "$(dirname "$0")/../env.sh"; . "$FEC_REPO/ch1/common.sh"
S11=$SCR/stage11_ch1; PK=$WS/stage11_run/ch1_package_final; LC=$WS/stage11_run/l_corr
echo "=== ch1 train profile=$CH1_PROFILE ($CH1_DESC) HEADS=$HEADS"
if [ "$HEADS" = v22c10 ]; then   # training cache of the v2.2 heads (CPU, a few minutes; skipped when it exists)
  [ -s "$HEAD22_FEAT/train/X.npy" ] || "$WS/.venv/bin/python" "$FEC_REPO/ch1/build_feat22.py" "$HEAD22_FEAT" ${FEAT22_X_SHA256:+--expect-sha256 $FEAT22_X_SHA256}
fi
if [ "${USE_SLURM:-0}" = 1 ]; then exec bash "$PK/train.sh" "$@"; fi
ROOT=${1:-$S11/final_path/train_$(date -u +%Y%m%dT%H%M%SZ)}
BASE_FEAT=${BASE_FEAT:-$S11/e_cx/h150/feat}; HEAD_FEAT=${HEAD_FEAT:-$S11/f_corr_canon/feat_all}
CONST=${CONST:-$S11/f_corr_canon/full/corrector/blend_frozen_gv21c.json}
mkdir -p "$ROOT"/{base,heads/logs,logs}; cp -n "$CONST" "$ROOT/heads/blend_frozen_gv21c.json"
export S11 PK C2=$WS/stage11_run/ch1d_fold2 CUBLAS_WORKSPACE_CONFIG=:4096:8 TMPDIR=${TMPDIR:-/tmp}
. "$WS/.venv/bin/activate"; T0=$(date +%s); NG=${NGPU:-1}
( export OMP_NUM_THREADS=8; D=$ROOT/base FEAT=$BASE_FEAT NGPU=$NG bash "$PK/pf_train_base.sh" ) 2>&1 | tee "$ROOT/logs/train_base.log"
if [ "$HEADS" = v22c10 ]; then
  mkdir -p "$ROOT/heads22/logs"; i=0
  for t in a b c d; do sd=$(seq -s, $(( i * 6 )) $(( i * 6 + 5 )))
    ( cd "$LC" && CUDA_VISIBLE_DEVICES=$(( i % NG )) CC_OUT=$ROOT/heads22 CC_FEAT=$HEAD22_FEAT LH_W=1024 LH_NB=8 LH_P=0.15 OMP_NUM_THREADS=8 \
        python l_hp.py train --tag c10$t --seeds "$sd" --epochs 42 --no-gru --full > "$ROOT/heads22/logs/heads_c10$t.log" 2>&1; echo "TRAIN_HEADS22 c10$t rc=$?" ) &
    i=$(( i + 1 )); [ $(( i % NG )) -ne 0 ] || wait
  done; wait
  echo "heads22: $(ls "$ROOT"/heads22/ck_c10[abcd]_s*.pt 2>/dev/null | wc -l) / 24"
else
  ( cd "$WS/stage11_run/ch1_corrector" && CC_OUT=$ROOT/heads CC_FEAT=$HEAD_FEAT OMP_NUM_THREADS=8 python cc_heads.py train --tag direct2 --seeds 0,1,2,3,4,5 --epochs 42 --no-gru --full > "$ROOT/heads/logs/heads_direct2.log" 2>&1; echo "TRAIN_HEADS rc=$? heads: $(ls "$ROOT"/heads/ck_direct2_s*.pt | wc -l) / 6" )
fi
if [ "${OOF:-0}" = 1 ]; then ( export CC_OUT=$ROOT/oof; mkdir -p "$CC_OUT/logs"; bash "$WS/stage11_run/ch1_corrector/job_full.sbatch" ) 2>&1 | tee "$ROOT/logs/train_oof.log"; fi
[ "${CMP:-0}" != 1 ] || python "$PK/pf_wcmp.py" "$ROOT" "$ROOT/wcmp.json"
echo "TRAIN_DONE root=$ROOT wall_s $(( $(date +%s) - T0 )) on $NG GPU(s)"
