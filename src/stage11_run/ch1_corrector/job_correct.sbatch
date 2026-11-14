#!/bin/bash
# Model-agnostic corrector as a job:  sbatch --export=ALL,PRED=<npz>,TAG=<name>[,SPLIT=test,FAM=gv21,TGT=rob] job_correct.sbatch
#SBATCH -J cc_correct
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH -t 02:30:00
#SBATCH -o slurm_%x_%j.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export CC_OUT=${CC_OUT:-$SCR/stage11_ch1/corrector}
export CC_FEAT=${CC_FEAT:-$SCR/stage11_ch1/feat_v21}
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=$CC_OUT/tmp; mkdir -p "$TMPDIR"
source "$WS/.venv/bin/activate"
echo "=== job $SLURM_JOB_ID on $(hostname) start $(date -u) TAG=${TAG:-} SPLIT=${SPLIT:-holdout}"
bash "$WS/stage11_run/ch1_corrector/run_correct.sh"
for f in "$CC_OUT"/scores/score_*${TAG}*.log; do [ -e "$f" ] || continue
  n=$(basename "$f" .log); n=${n#score_}
  echo "OFFICIAL $n S $(grep -h COMPOSITE "$f" | sed 's/.*S = //;s/ .*//') psi $(grep -h 'R2_psi' "$f" | awk '{print $2}') 1-D $(grep -h '1 - D_LCFS' "$f" | awk '{print $4}') Cons $(grep -h 'Consistency ' "$f" | awk '{print $2}') | $(grep -hE '^\s+(R_axis|Z_axis|kappa|tri_top|tri_bot|volume|li)\s' "$f" | awk '{printf "%s ", $2}')"
done
echo "=== end $(date -u)"
