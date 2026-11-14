# DIII-D TRAIN, MAST-like pseudo-horizontal machinery (clean), 300 train shots x 8 frames (stride 23, holdout excluded):
#   kinetic-scalar self-calibration (C_q95 = the pipeline's own DIII-D calibration, keepCq convention), scal2 c on THIS pipeline's maps
#   (core-covered frames = MAST coverage), q95 gap attribution.  $1 = tag, rest = run_inv7 args
set -uo pipefail
TAG=$1; shift
O=$OUT7/final; mkdir -p $O
$PY run_inv7.py --machine d3d --mode pseudo --data-dir $DATA/diii_d_train --exclude-holdout $HOLD --stride 23 --wall $GEOM/d3d_limiter_mask.npz \
   --cfg $G1/configs/cfg_soft_k50.json --cfg $R7/configs/inv7_base.json "$@" --set "stress={}" --tag ${TAG}_pcal --out $O --frames-per-shot 8 --procs $NPROC 2>&1 | grep "SUMMARY\|Traceback\|Error" | cut -c1-2000
$PYS $G1/eval_frames.py --run $O --tag ${TAG}_pcal --data-dir $DATA/diii_d_train --self-calib --procs 48 2>&1 | grep "EVAL\|gauge\|Error\|Traceback"
cp $O/calib_${TAG}_pcal.json $O/calib_${TAG}.json
$PY $SCALC/scal_frames.py --machine d3d --src g1 --run $O --tag ${TAG}_pcal --data-dir $DATA/diii_d_train --wall $GEOM/d3d_limiter_mask.npz --out $O/scal_d3d_${TAG}_pcal.parquet --procs $NPROC | grep SUMMARY
$PYS $INV6/scal_c.py $O/scal_d3d_${TAG}_pcal.parquet $O/c_${TAG}_mast.json --use core | cut -c1-1500
$PYS $INV6/diag_kappa_d3d.py $O ${TAG}_pcal 2>&1 | tail -40
