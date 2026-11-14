#!/bin/bash
# ch1_package_final — ONE command that retrains every learned member the FcCX predict path loads, from scratch, on ALL 7,041 labelled
# DIII-D shots (train 6,891 + holdout 150), fixed epochs and seeds, no validation, no early stopping:
#
#   bash train.sh [MODEL_ROOT]        (default $SCR/stage11_ch1/final_path/train_<UTC stamp>; never the shipped roots)
#
# Submits 2 GPU jobs on workq (they do not depend on each other) and prints their ids:
#   base   (4 GPU)  pf_train_base.sh : FM 11 nets + TW 4 seeds + SC 17 members on the mirror-canonical FEATURE-V2.2 cache   -> $ROOT/base
#   heads  (1 GPU)  ch1_corrector/cc_heads.py train --full : canonical direct2 heads, residual MLP 768x4, 6 seeds x 42 epochs,
#                   on the mirror-canonical FEATURE-V2.1 cache (f_corr_canon/feat_all)                                       -> $ROOT/heads
#   OOF=1 adds      ch1_corrector/job_full.sbatch (B1): the gv21 OOF teacher statistics cov_oof / steps / dsc (K=4 x 2 seeds + 4 full
#                   seeds, 30 epochs; 0.20 GH200 GPU-h measured, job 7277427)                                                -> $ROOT/oof
# Afterwards:  BASE_ROOT=$ROOT/base HEAD_ROOT=$ROOT/heads HEAD22_ROOT=$ROOT/heads22 [OOF_ROOT=$ROOT/oof] bash predict.sh <shot_dir> <name> [order]
#   (HEADS=v21 keeps the FcCX v2.1 heads job and skips the v2.2 heads; HEAD_ROOT then only supplies blend_frozen_gv21c.json)
# Prerequisites = generated data and frozen statistics (CPU; produced by the scripts named below; listed with sha256 in weights/GENERATED.sha256, see docs/CH1_GENERATED_DATA.md):
#   FEATURE-V2.1 / V2.2 caches (ch1_features/prep_v21.py, e_ip_axis/prep_v22.py), mirror-canonical caches + TRAIN scalers
#   (e_fold2_offset/e_canon.py VARIANT=cfl on v2.1 -> e_fold2_offset/h150_cfl/feat ; e_cx/cx_canon.py on v2.2 -> e_cx/h150/feat ;
#   f_corr_canon/f_merge.py -> f_corr_canon/feat_all), PCA bases stage6_H/pca_{80,120}.npz, metric stage11_ch1/metric (L_80, L_120, M_80),
#   gv21 OOF statistics (corrector_full, unless OOF=1), and the frozen rob constants blend_frozen_gv21c.json (33 numbers fitted on a
#   344-shot validation slice of the 6,891-shot dev run with dev-style canonical heads: f_corr_canon/f_heads.sh, h150 part).
set -euo pipefail
WS=${WS:?set WS}
SCR=${SCR:?set SCR}
S11=$SCR/stage11_ch1; PK=$WS/stage11_run/ch1_package_final
ROOT=${1:-$S11/final_path/train_$(date -u +%Y%m%dT%H%M%SZ)}
RR=$(readlink -m "$ROOT")
for d in "$S11/e_cx" "$S11/f_corr_canon" "$S11/corrector_full" "$S11/e_fold2_offset" "$S11/corrector"; do
  case "$RR/" in "$(readlink -f "$d")"/*) echo "refusing to write into the shipped root $d"; exit 1;; esac
done
BASE_FEAT=${BASE_FEAT:-$S11/e_cx/h150/feat}; HEAD_FEAT=${HEAD_FEAT:-$S11/f_corr_canon/feat_all}
CONST=${CONST:-$S11/f_corr_canon/full/corrector/blend_frozen_gv21c.json}
export TMPDIR=/tmp
mkdir -p "$ROOT"/{base,heads/logs,logs}
cp -n "$CONST" "$ROOT/heads/blend_frozen_gv21c.json"
COMMON="export WS=$WS SCR=$SCR S11=$S11 PK=$PK C2=$WS/stage11_run/ch1d_fold2 PYTHONUNBUFFERED=1 MPLCONFIGDIR=$S11/.mplcache XDG_CACHE_HOME=$S11/xdg_cache CUBLAS_WORKSPACE_CONFIG=:4096:8; export TMPDIR=/tmp/pf_\$SLURM_JOB_ID; mkdir -p \$TMPDIR; . $WS/.venv/bin/activate; nvidia-smi -L | head -4"
J1=$(sbatch --parsable -J pf_tr_base -p workq -N 1 --gpus=4 --cpus-per-task=288 --mem=440G -t 01:30:00 -o "$ROOT/logs/slurm_%x_%j.out" \
  --wrap "$COMMON; export OMP_NUM_THREADS=8; D=$ROOT/base FEAT=$BASE_FEAT bash $PK/pf_train_base.sh")
J2=$(sbatch --parsable -J pf_tr_heads -p workq -N 1 --gpus=1 --cpus-per-task=72 --mem=200G -t 01:00:00 -o "$ROOT/logs/slurm_%x_%j.out" \
  --wrap "$COMMON; T0=\$(date +%s); cd $WS/stage11_run/ch1_corrector && CC_OUT=$ROOT/heads CC_FEAT=$HEAD_FEAT OMP_NUM_THREADS=8 python cc_heads.py train --tag direct2 --seeds 0,1,2,3,4,5 --epochs 42 --no-gru --full > $ROOT/heads/logs/heads_direct2.log 2>&1; echo \"TRAIN_HEADS rc=\$? wall_s \$(( \$(date +%s) - T0 )) x 1 GPU\"; ls $ROOT/heads/ck_direct2_s*.pt | wc -l")
if [ "${HEADS:-v22c10}" = v22c10 ]; then     # the v2.2 c10 heads replace the v2.1 direct2 heads (4 GPU jobs x 6 seeds)
  scancel "$J2" 2>/dev/null; J2=""
  HEAD22_FEAT=${HEAD22_FEAT:-$S11/l_corr/feat22_all}; LC=$WS/stage11_run/l_corr
  [ -s "$HEAD22_FEAT/train/X.npy" ] || { echo "build the FULL v2.2 head cache first: python ch1/build_feat22.py (ch1/train.sh does it)"; exit 1; }
  mkdir -p "$ROOT/heads22/logs"
  for t in a b c d; do i=$(printf %d "'$t"); i=$(( i - 97 )); sd=$(seq -s, $(( i * 6 )) $(( i * 6 + 5 )))
    J2="$J2 $(sbatch --parsable -J pf_tr_h22$t -p workq -N 1 --gpus=1 --cpus-per-task=72 --mem=200G -t 01:00:00 -o "$ROOT/logs/slurm_%x_%j.out" \
      --wrap "$COMMON; cd $LC && CC_OUT=$ROOT/heads22 CC_FEAT=$HEAD22_FEAT LH_W=1024 LH_NB=8 LH_P=0.15 OMP_NUM_THREADS=8 python l_hp.py train --tag c10$t --seeds $sd --epochs 42 --no-gru --full > $ROOT/heads22/logs/heads_c10$t.log 2>&1; echo TRAIN_HEADS22 rc=\$?")"
  done
fi
J=$(echo $J1 $J2)        # single blanks (J2 is empty or starts with a blank when the v2.2 heads replace the v2.1 job)
if [ "${OOF:-0}" = 1 ]; then
  mkdir -p "$ROOT/oof/logs"
  J="$J $(cd "$WS/stage11_run/ch1_corrector" && sbatch --parsable -J pf_tr_oof -o "$ROOT/logs/slurm_%x_%j.out" --export=ALL,CC_OUT=$ROOT/oof job_full.sbatch)"
fi
if [ "${CMP:-0}" = 1 ]; then   # compare the new weights with the shipped FcCX weights (sha256, then tensor equality)
  JC=$(sbatch --parsable -J pf_tr_cmp -p workq -N 1 --cpus-per-task=16 --mem=60G -t 00:30:00 --dependency=afterany:$(echo $J1 $J2 | tr -s " " ":") -o "$ROOT/logs/slurm_%x_%j.out" \
    --wrap ". $WS/.venv/bin/activate; export SCR=$SCR HEADS=${HEADS:-v22c10}; python $PK/pf_wcmp.py $ROOT $ROOT/wcmp.json")
  echo "weight comparison job $JC -> $ROOT/wcmp.json"
fi
echo "$J" > "$ROOT/train_jobs.txt"
echo "TRAIN_SUBMITTED root=$ROOT jobs: $J"
echo "GPU-h when done: sacct -X -n -j ${J// /,} --format=JobID,JobName,Elapsed,AllocTRES%50"
