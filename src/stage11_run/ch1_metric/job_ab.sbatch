#!/bin/bash
# Stage 11 / ch1_metric — loss A/B (1 GPU + 72 cores): torch variants on GPU, sklearn incumbent-recipe fit on CPU,
# then the official harness on every prediction file.
#SBATCH -J metric_ab
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH -t 01:30:00
#SBATCH -o slurm_%x_%j.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export OUT=${OUT:-$SCR/stage11_ch1/metric}
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
source "$WS/.venv/bin/activate"
cd "$WS/stage11_run/ch1_metric"
mkdir -p "$OUT/ab/scores"
echo "=== job $SLURM_JOB_ID on $(hostname) start $(date -u)"; nvidia-smi -L | head -2
SKPID=""
if [ "${SKIP_SK:-0}" != "1" ]; then THREADS=32 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 python ab_sklearn.py > "$OUT/ab/ab_sklearn.log" 2>&1 & SKPID=$!; fi
OMP_NUM_THREADS=8 python ab_train.py || echo AB_TRAIN_FAILED
score() {
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  for name in "$@"; do
    lg=$OUT/ab/scores/score_$name.log
    grep -q COMPOSITE "$lg" 2>/dev/null && continue
    python "$WS/stage2_run/score_stage2.py" --machine d3d --gt-dir "$WS/scaleup_run/data/diii_d_train" \
      --skip 1000 --n 150 --pred "$OUT/ab/preds_d3d_$name.npz" --tag "$name" > "$lg" 2>&1 &
  done
}
score $(cat "$OUT/ab/names.txt")
if [ -n "$SKPID" ]; then wait $SKPID; tail -2 "$OUT/ab/ab_sklearn.log"; score ab_skG_s42; fi
wait
for f in "$OUT"/ab/scores/score_*.log; do
  n=$(basename $f .log); n=${n#score_}
  echo "$n $(grep -h COMPOSITE $f | sed 's/.*S = //;s/ .*//') psi $(grep -h 'R2_psi' $f | awk '{print $2}') 1-D $(grep -h '1 - D_LCFS' $f | awk '{print $4}') Cons $(grep -h 'Consistency ' $f | awk '{print $2}') | $(grep -hE '^\s+(R_axis|Z_axis|kappa|tri_top|tri_bot|volume|li)\s' $f | awk '{printf "%s ", $2}')"
done
echo "=== end $(date -u)"
