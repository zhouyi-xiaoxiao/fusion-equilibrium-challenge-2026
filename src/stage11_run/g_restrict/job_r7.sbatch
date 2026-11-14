#!/bin/bash
# Generic g_restrict job wrapper: sbatch [--cpus-per-task=N -t HH:MM:SS -J name] job_r7.sbatch steps/<step>.sh [args]
#SBATCH -J r7
#SBATCH -p workq
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH -o slurm_%x_%j.out
set -uo pipefail
export WS=${WS:?set WS}
export SCR=${SCR:?set SCR}
export G1=$WS/stage11_run/ch2_G1
export INV=$WS/stage11_run/ch2d_inverse
export INV6=$WS/stage11_run/e_mast_inv6
export R7=$WS/stage11_run/g_restrict
export SCALC=$WS/stage11_run/ch2d_scalars
export OUT=$SCR/stage11_ch2/inverse
export OUT6=$SCR/stage11_ch2/inv6
export OUT7=$SCR/stage11_ch2/g_restrict
export GEOM=$SCR/stage11_ch2_geometry
export DATA=$SCR/data_full
export HOLD=$WS/scaleup_run/data/diii_d_train:1000:1150
export PY=$WS/.venv_gs/bin/python
export PYS=$WS/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=$OUT7/.mplcache
export NPROC=${SLURM_CPUS_PER_TASK:-8}
mkdir -p $OUT7/logs $OUT7/dev $OUT7/mast_dev $OUT7/final $MPLCONFIGDIR
echo "=== job ${SLURM_JOB_ID:-local} $(hostname) $(date -u) step: $*"
t0=$(date +%s)
cd $R7
bash "$@"
rc=$?
echo "=== rc=$rc WALL_SECONDS $(( $(date +%s) - t0 )) on $NPROC cores (CPU-only)"
exit $rc
