#!/bin/bash
# Stage 11 / ch1_metric — FD Jacobian shards (CPU only). Usage:
#   sbatch --array=0-7 job_fd.sbatch            (full run, NSHARD=8)
#   sbatch --array=0 --export=ALL,MAXF=70 -t 00:15:00 job_fd.sbatch   (smoke)
#SBATCH -J metric_fd
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH -t 01:30:00
#SBATCH -o slurm_%x_%A_%a.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export OUT=${OUT:-$SCR/stage11_ch1/metric}
export SHARD=${SLURM_ARRAY_TASK_ID:-0}
export NSHARD=${NSHARD:-8}
export WORKERS=${WORKERS:-70}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
mkdir -p "$OUT/logs" "$OUT/fd"
source "$WS/.venv/bin/activate"
cd "$WS/stage11_run/ch1_metric"
echo "=== job $SLURM_JOB_ID task $SHARD on $(hostname) cpus=$SLURM_CPUS_PER_TASK start $(date -u)"
python fd_worker.py
rc=$?
echo "=== end $(date -u) rc=$rc"
exit $rc
