# Package a full MAST run (keepCq convention): q95 = contour integral x the pipeline's OWN DIII-D pseudo-train calibration C_q95
#   (calib_<tag>.json of r30_pcal), validity window + med3; beta_N = scal2 with c of r30_pcal (core-covered).  float32 psi (24-bit
#   mantissa as INV_P5/P6), HF stream order asserted.  Then validate_local, official derivation pass/fail (QA only), official
#   validator on ALL 1,206 stream shots, label-free isotherm/edge evaluation, exported-map current check.  $1 = tag, $2 = name
set -uo pipefail
TAG=$1; NAME=$2
O=${MAST_OUT:-$OUT7/final}; MD=${MAST_DIR:-$DATA/mast}
ORDER=${ORDER:-$SCR/stage8_P_ops_mast_package/hub_stream_order_mast_public_test.txt}
CT=${CTAG:-$TAG}
[ -f $OUT7/final/c_${CT}_mast.json ] && [ -f $OUT7/final/calib_${CT}.json ] || { echo "missing DIII-D calibration for $CT"; exit 2; }
CQ=$(python3 -c "import json; print(repr(json.load(open('$OUT7/final/calib_${CT}.json'))['C_q95']))")
echo "C_q95 (own DIII-D pseudo-train calibration) = $CQ ; c_betaN json: $(python3 -c "import json; print(json.load(open('$OUT7/final/c_${CT}_mast.json'))['c_used'])")"
OA=""; [ "$ORDER" != none ] && OA="--order-file $ORDER"
S=$O/submission/$NAME; mkdir -p $S
[ -f $S/mast_public_test.npz ] && { echo "refusing to overwrite $S/mast_public_test.npz"; exit 3; }
$PY $INV6/pack_inv6.py --run $O --tag $TAG --machine mast --scal $O/scal_mast_$TAG.parquet --c-json $OUT7/final/c_${CT}_mast.json $OA --psi-bits 24 --q-factor $CQ --out $S/mast_public_test.npz | cut -c1-1500 || exit 1
$PYS $WS/stage6_run/variants/P_ops_mast_package/validate_local.py $S/mast_public_test.npz --shot-dir $MD > $S/validate_local.log 2>&1; echo "validate_local rc=$?"; tail -2 $S/validate_local.log
$PYS $G1/lcfs_check.py $S/mast_public_test.npz $S/lcfs_check.json --run $O --tag $TAG > /dev/null 2>&1; echo "lcfs_check rc=$?"
python3 -c "import json; d=json.load(open('$S/lcfs_check.json')); print('LCFSCHECK (pass/fail QA only) success', d['derivation_success_frac'], 'fallback', d.get('fallback'))"
md5sum $S/mast_public_test.npz | tee $S/md5.txt; ls -la $S/mast_public_test.npz
export HF_HOME=$SCR/stage8_P_ops_mast_package/hf_home
(cd $S && timeout 3000 $PYS $WS/starter/validate_submission.py mast_public_test.npz --config mast_public_test --max-shots 0 > validate_official_all.log 2>&1; echo "official validator (ALL stream shots) rc=$?"; tail -2 validate_official_all.log)
md5sum $S/mast_public_test.npz
$PY $INV/iso_eval.py --run $O --tag $TAG --data-dir $MD --wall $GEOM/mast_limiter_mask.npz --cfg $G1/configs/cfg_soft_k50.json --stride 6 --procs 96 --out $O/isoeval_${TAG}_s6.json 2>&1 | grep "ISOEVAL\|Error\|Traceback"
$PY $INV/map_current_check.py $S/mast_public_test.npz --data-dir $MD --wall $GEOM/mast_limiter_mask.npz --every 30 2>&1 | grep "MAPCURRENT\|Error\|Trace"
