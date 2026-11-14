#!/bin/bash
# Stage 11 / ch1_metric — per-frame Jacobians for a whole split (CPU only).
#   sbatch --array=0-19 job_jfull.sbatch                                   (train, K=80, NSH=20)
#   sbatch --array=0 --export=ALL,SPLIT=holdout,NSH=1 job_jfull.sbatch     (holdout)
#SBATCH -J metric_jfull
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=60G
#SBATCH -t 01:30:00
#SBATCH -o slurm_%x_%A_%a.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export OUT=${OUT:-$SCR/stage11_ch1/metric}
export SHARD=${SLURM_ARRAY_TASK_ID:-0}
export SPLIT=${SPLIT:-train} K=${K:-80} DMULT=${DMULT:-1.0} NSH=${NSH:-20} WORKERS=${WORKERS:-70}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=$SCR/stage11_ch1/.mplcache XDG_CACHE_HOME=$SCR/stage11_ch1/xdg_cache
source "$WS/.venv/bin/activate"
cd "$WS/stage11_run/ch1_metric"
echo "=== job $SLURM_JOB_ID task $SHARD on $(hostname) SPLIT=$SPLIT K=$K DMULT=$DMULT NSH=$NSH start $(date -u)"
python fd_full.py
rc=$?
echo "=== end $(date -u) rc=$rc"
exit $rc
