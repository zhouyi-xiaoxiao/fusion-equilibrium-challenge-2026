# MAST pipeline profile "p7a_tsw05" = INV_P7a_tsw05 (mast_public_test.npz md5 c052f31c…): the p7a pipeline followed by one
# post-processing stage, temporal smoothing of the plasma part of every map along the shot (random-walk / local-level Kalman-RTS
# smoother, bandwidth 0.5 frame; fallback / invalid / filled frames are missing observations; the vacuum flux of every frame is
# exact; q95 and beta_N are copied unchanged). The bandwidth is a fixed constant (no fit on the shots being predicted).
. "$FEC_REPO/ch2/pipelines/p7a.sh"
PIPE_DESC="INV_P7a + temporal smoothing of the plasma flux (sigma 0.5 frame)"
CONST_PIPE=p7a                                      # q95 / beta_N constants of the base pipeline (scalars are not smoothed)
POST=(tsmooth)
REF_MD5=c052f31c99426496da89ff5adfe0a2bb
post_tsmooth() {  # <run dir> <input npz> <output npz> <mast shot dir>
  local O=$1 IN=$2 OUTF=$3 MD=$4 M=$1/post_tsmooth
  ( cd "$WS/stage11_run/j_temporal" && $PY tsmooth.py --machine mast --run "$O" --tag "$TAG" --data-dir "$MD" --wall "$GEOM/mast_limiter_mask.npz" \
      --cfg "$O/cfg_${TAG}_mast.json" --out "$M" --specs "$FEC_REPO/ch2/pipelines/tsmooth_w05.json" --only w05 --procs "$NPROC" ) 2>&1 | grep -E "TSMOOTH|Error|Traceback" | cut -c1-600
  ( cd "$WS/stage11_run/j_temporal" && $PY pack_ts2.py --src "$IN" --base-run "$O" --base-tag "$TAG" --run "$M" --tag "${TAG}_w05" --out "$OUTF" \
      --psi-bits 24 --what "$PIPE" ) 2>&1 | cut -c1-600
  [ -s "$OUTF" ]
}
