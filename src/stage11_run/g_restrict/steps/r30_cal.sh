# DIII-D TRAIN real chords (300 train x 8 frames): kinetic-scalar self-calibration + scal2 c for the DIII-D coverage (all frames)
set -uo pipefail
TAG=$1; shift
O=$OUT7/final; mkdir -p $O
$PY run_inv7.py --machine d3d --mode real --data-dir $DATA/diii_d_train --exclude-holdout $HOLD --stride 23 --wall $GEOM/d3d_limiter_mask.npz \
   --cfg $G1/configs/cfg_soft_k50.json --cfg $R7/configs/inv7_base.json "$@" --tag ${TAG}_cal --out $O --frames-per-shot 8 --procs $NPROC 2>&1 | grep "SUMMARY\|Traceback\|Error" | cut -c1-2000
$PYS $G1/eval_frames.py --run $O --tag ${TAG}_cal --data-dir $DATA/diii_d_train --self-calib --procs 48 2>&1 | grep "EVAL\|gauge\|Error\|Traceback"
cp $O/calib_${TAG}_cal.json $O/calib_${TAG}_real.json
$PY $SCALC/scal_frames.py --machine d3d --src g1 --run $O --tag ${TAG}_cal --data-dir $DATA/diii_d_train --wall $GEOM/d3d_limiter_mask.npz --out $O/scal_d3d_${TAG}_cal.parquet --procs $NPROC | grep SUMMARY
$PYS $INV6/scal_c.py $O/scal_d3d_${TAG}_cal.parquet $O/c_${TAG}_d3d.json --use all | cut -c1-1500
