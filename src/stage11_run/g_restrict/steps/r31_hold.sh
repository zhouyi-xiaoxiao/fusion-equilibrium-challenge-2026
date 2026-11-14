# DIII-D 150 HOLDOUT shots, ALL frames, real chords, OFFICIAL harness (score_stage2): (a) kinetic scalars x the real-chord train
#   calibration, (b) the shipped scalar method (keepCq: q95 x C_q95 of the pseudo calibration in the MAST file; scal2 beta_N with the DIII-D-coverage c of r30_cal).
set -uo pipefail
TAG=$1; shift
O=$OUT7/final; mkdir -p $O
$PY run_inv7.py --machine d3d --mode real --data-dir $DATA/diii_d_train --holdout-names $HOLD --wall $GEOM/d3d_limiter_mask.npz \
   --cfg $G1/configs/cfg_soft_k50.json --cfg $R7/configs/inv7_base.json "$@" --tag $TAG --out $O --preds --procs $NPROC 2>&1 | grep "SUMMARY\|Traceback\|Error" | cut -c1-2000
for i in $(seq 240); do [ -f $O/c_${TAG}_d3d.json ] && [ -f $O/calib_${TAG}_real.json ] && [ -f $O/calib_$TAG.json ] && break; sleep 15; done
$PY - <<PY
import json, numpy as np
c = json.load(open("$O/calib_${TAG}_real.json"))
z = dict(np.load("$O/preds_d3d_${TAG}_f32.npz"))
for k in list(z):
    if k.endswith("_q95"): z[k] = np.clip(c["C_q95"] * z[k], 1.2, 25).astype(np.float32)
    if k.endswith("_betaN"): z[k] = np.clip(c["C_betaN"] * z[k], 0, 8).astype(np.float32)
np.savez("$O/preds_d3d_${TAG}_kincal_f32.npz", **z)
print("kinetic scalars x train calib", c["C_q95"], c["C_betaN"])
PY
$PY $SCALC/scal_frames.py --machine d3d --src g1 --run $O --tag $TAG --data-dir $DATA/diii_d_train --wall $GEOM/d3d_limiter_mask.npz --out $O/scal_d3d_${TAG}.parquet --procs $NPROC | grep SUMMARY
CQ=$(python3 -c "import json; print(repr(json.load(open('$O/calib_$TAG.json'))['C_q95']))")
$PYS $INV6/pack_inv6.py --run $O --tag $TAG --machine d3d --scal $O/scal_d3d_${TAG}.parquet --c-json $O/c_${TAG}_d3d.json --q-factor $CQ --psi-bits 0 --out $O/preds_d3d_${TAG}_shipped_f32.npz | cut -c1-600
cd $O
for v in kincal_f32 shipped_f32; do
  $PYS $WS/stage2_run/score_stage2.py --machine d3d --gt-dir $WS/scaleup_run/data/diii_d_train --skip 1000 --n 150 --pred $O/preds_d3d_${TAG}_$v.npz --tag R7_${TAG}_$v > $O/score_d3d_R7_${TAG}_$v.log 2>&1 &
done
wait
for v in kincal_f32 shipped_f32; do echo "== $v"; grep -v "^\s*$" $O/score_d3d_R7_${TAG}_$v.log | tail -19 | head -17; done
cd $R7; $PY $INV/inv_summary.py $O $TAG | head -14
