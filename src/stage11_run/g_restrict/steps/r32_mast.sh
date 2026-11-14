# FULL MAST solve (label-free) + scal2 table.  $1 = tag, rest = run_inv7 args.  Packaging is a separate step (r33) so that the
#   file is only built after the pilot has been judged sane.
set -uo pipefail
TAG=$1; shift
O=${MAST_OUT:-$OUT7/final}; mkdir -p $O
MD=${MAST_DIR:-$DATA/mast}
$PY run_inv7.py --machine mast --mode real --data-dir $MD --wall $GEOM/mast_limiter_mask.npz --cfg $G1/configs/cfg_soft_k50.json --cfg $R7/configs/inv7_base.json "$@" \
   --tag $TAG --out $O --procs $NPROC 2>&1 | grep "SUMMARY\|Traceback\|Error" | cut -c1-2500
$PY $INV/inv_summary.py $O $TAG
$PY $SCALC/scal_frames.py --machine mast --src g1 --run $O --tag $TAG --data-dir $MD --wall $GEOM/mast_limiter_mask.npz --out $O/scal_mast_$TAG.parquet --procs $NPROC | grep SUMMARY
