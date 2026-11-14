# DIII-D 150 HOLDOUT shots x 8 frames with the MAST-like machinery (clean | stress), official functionals (eval_frames):
#   (a) kinetic scalars x the pseudo calibration of r30 (INV_P5/P6-comparable), (b) the SHIPPED scalar method = keepCq convention
#   (q95 = contour integral x C_q95 of r30_pcal, validity window, med3; scal2 beta_N with c from r30 train maps; no holdout fitting).
#   $1 = tag, $2 = condition, rest = run_inv7 args
set -uo pipefail
TAG=$1; C=$2; shift 2
if [ "$C" = clean ]; then ST='{}'; else ST='{"noise":0.1,"branch":0.05,"timing":10.0,"sep":0.03,"edge_noise":0.01}'; fi
O=$OUT7/final; mkdir -p $O
$PY run_inv7.py --machine d3d --mode pseudo --data-dir $DATA/diii_d_train --holdout-names $HOLD --wall $GEOM/d3d_limiter_mask.npz \
   --cfg $G1/configs/cfg_soft_k50.json --cfg $R7/configs/inv7_base.json "$@" --set "stress=$ST" --tag ${TAG}_ph$C --out $O --frames-per-shot 8 --procs $NPROC 2>&1 | grep "SUMMARY\|Traceback\|Error" | cut -c1-2000
for i in $(seq 240); do [ -f $O/c_${TAG}_mast.json ] && [ -f $O/calib_$TAG.json ] && break; sleep 15; done
$PYS $G1/eval_frames.py --run $O --tag ${TAG}_ph$C --data-dir $DATA/diii_d_train --calib $O/calib_$TAG.json --procs 48 2>&1 | grep "EVAL\|gauge\|Error\|Traceback"
$PY $SCALC/scal_frames.py --machine d3d --src g1 --run $O --tag ${TAG}_ph$C --data-dir $DATA/diii_d_train --wall $GEOM/d3d_limiter_mask.npz --out $O/scal_d3d_${TAG}_ph$C.parquet --procs $NPROC | grep SUMMARY
CQ=$(python3 -c "import json; print(repr(json.load(open('$O/calib_$TAG.json'))['C_q95']))")
$PYS $INV6/pack_inv6.py --run $O --tag ${TAG}_ph$C --machine d3d --scal $O/scal_d3d_${TAG}_ph$C.parquet --c-json $O/c_${TAG}_mast.json --q-factor $CQ --parts-out $O/scal2_${TAG}_ph$C | cut -c1-600
$PYS $G1/eval_frames.py --run $O/scal2_${TAG}_ph$C --tag ${TAG}_ph$C --data-dir $DATA/diii_d_train --calib $INV6/configs/calib_unity.json --procs 48 2>&1 | grep "EVAL\|Error\|Traceback" | sed 's/^/[shipped scalars keepCq] /'
$PYS $G1/eval_frames.py --run $O/scal2_${TAG}_ph$C --tag ${TAG}_ph$C --data-dir $DATA/diii_d_train --calib $INV6/configs/calib_unity.json --f32 --procs 48 2>&1 | grep "EVAL" | sed 's/^/[shipped scalars keepCq f32] /'
