#!/bin/bash
# Stage 11 / ch1_metric — build + holdout sanity (CPU only). Usage: sbatch [--export=ALL,STEPS=build] job_post.sbatch
#SBATCH -J metric_post
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o slurm_%x_%j.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export OUT=${OUT:-$SCR/stage11_ch1/metric}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
export MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
source "$WS/.venv/bin/activate"
cd "$WS/stage11_run/ch1_metric"
echo "=== job $SLURM_JOB_ID on $(hostname) cpus=$SLURM_CPUS_PER_TASK start $(date -u) STEPS=${STEPS:-all}"
bash ./run_post.sh
rc=$?
echo "=== end $(date -u) rc=$rc"
exit $rc
