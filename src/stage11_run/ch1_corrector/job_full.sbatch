#!/bin/bash
# FINAL REFIT of the consistency corrector on ALL 7,041 labelled DIII-D shots (train 6,891 + holdout 150), fixed epochs
# and seeds, then inference on any feature directory through the generic wrapper (here: the 874-shot public test).
#   B1 (GPU, 13 min measured):  sbatch job_full.sbatch          (STAGES="merge base heads derive dsc"; outputs $SCR/stage11_ch1/corrector_full)
#   B2 (CPU only), any feature dir under $CC_FEAT (e.g. test, or a build_unseen.py output dir):
#     sbatch --dependency=afterok:<B1> --export=ALL,CC_FEAT=$SCR/stage11_ch1/feat_v21_all,CC_OUT=$SCR/stage11_ch1/corrector_full,\
#       PRED=<any base harness npz for that split>|COEF=<coef npy>,QB=<npz with q95/betaN>,TAG=<name>,SPLIT=test,FAM=gv21,TGT=rob,\
#       BLEND_FROZEN=blend_frozen_gv21.json,DSC=dsc_gv21.json job_correct.sbatch
#   (STAGES=test inside this job does the same with the gv21 base and $QB)
# Recipe (all fixed, no early stopping, no holdout-based choice):
#   base (OOF teacher, family statistics + optional base psi): cc_base.py  MLP D-1024-1024-512-80, metric loss L_80,
#        AdamW 1e-3 OneCycle 30 epochs batch 4096; K=4 shot folds x OOF seeds {0,1}; full fits seeds {0,1,2,3}
#   direct scalar heads: cc_heads.py --full  residual MLP 768x4, 42 epochs cosine, EMA 0.995, seeds {0..5}
#   frozen constants from the 6,891-shot dev run: blend_holdout_gv21.json (rob clip/lambda per scalar)
#   GN step: r=1e9 (no trust clip), eta=1, iters=13 (1 + 3 frozen-J steps, re-linearise once if residual > 0.3 prior RMS)
#SBATCH -J cc_full
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=230G
#SBATCH -t 03:30:00
#SBATCH -o slurm_%x_%j.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
SRC_FEAT=$SCR/stage11_ch1/feat_v21
DEV_OUT=$SCR/stage11_ch1/corrector
export CC_FEAT=${CC_FEAT:-$SCR/stage11_ch1/feat_v21_all}
export CC_OUT=${CC_OUT:-$SCR/stage11_ch1/corrector_full}
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$CC_OUT/logs" "$CC_OUT/scores"; export TMPDIR=$CC_OUT/tmp; mkdir -p "$TMPDIR"
source "$WS/.venv/bin/activate"
cd "$WS/stage11_run/ch1_corrector"
L=$CC_OUT/logs; NW=64
STAGES=${STAGES:-merge base heads derive dsc}
QB=${QB:-$SCR/stage8_L_ch1_finish_package/submission/diii_d_public_test.npz}
has() { [[ " $STAGES " == *" $1 "* ]]; }
echo "=== job ${SLURM_JOB_ID:-} on $(hostname) start $(date -u) STAGES=$STAGES CC_FEAT=$CC_FEAT CC_OUT=$CC_OUT"
T0=$(date +%s); el() { echo "--- [$(( $(date +%s) - T0 ))s] $*"; }
if has merge; then
  [ -e "$CC_FEAT/train/X.npy" ] || python cc_merge_feat.py --src "$SRC_FEAT" --dst "$CC_FEAT" --splits train,holdout || { echo MERGE_FAILED; exit 1; }
  el merge
fi
cp -n "$DEV_OUT/blend_holdout_gv21.json" "$CC_OUT/blend_frozen_gv21.json"
[ -e "$DEV_OUT/hold_ref.npz" ] && cp -n "$DEV_OUT/hold_ref.npz" "$CC_OUT/"
if has base; then
  TAG=gv21 KF=4 OOF_SEEDS=0,1 FULL_SEEDS=0,1,2,3 EPOCHS=30 LOSS=G OMP_NUM_THREADS=8 python cc_base.py > "$L/base.log" 2>&1 || { echo BASE_FAILED; tail -5 "$L/base.log"; exit 1; }
  tail -1 "$L/base.log"; python cc_cov.py --coef "$CC_OUT/coef_oof_gv21.npy" --tag gv21
  el base
fi
if has heads; then
  ( OMP_NUM_THREADS=8 python cc_heads.py train --tag direct2 --seeds 0,1,2,3,4,5 --epochs 42 --no-gru --full > "$L/heads_direct2.log" 2>&1; echo "rc=$?" >> "$L/heads_direct2.log" ) &
fi
if has derive; then
  python cc_derive.py --coef "$CC_OUT/coef_oof_gv21.npy" --tag gv21 --split train --workers $NW > "$L/derive_gv21_train.log" 2>&1 || echo DERIVE_FAILED
  el derive
fi
wait
has heads && { grep "seed\|rc=" "$L/heads_direct2.log" | cut -c1-200; el heads; }
has dsc && python cc_dsc.py --base-train f11_train_gv21.npy --tag gv21
if has test; then
  COEF=$CC_OUT/coef_test_gv21.npy QB=$QB TAG=gv21 SPLIT=test FAM=gv21 TGT=rob BLEND_FROZEN=blend_frozen_gv21.json \
    DSC=dsc_gv21.json bash run_correct.sh 2>&1 | tail -3 | cut -c1-300
  el test
fi
echo "=== end $(date -u)"
