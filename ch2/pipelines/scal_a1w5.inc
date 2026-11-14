# Post stage "a1w5" (scalar rules A1 + B7w5) of the profile dec_e4s_c1nx_a1w5. Not a profile: it is sourced AFTER corr_b50.inc and
# rewrites only q95 / beta_N of the corrected file; psi stays bit for bit that of the corrected file.
#   A1 (q95):    q95 = q95_blend x H95[corrected map] / H95[decoded, uncorrected map], H95 = q at psi_N 0.95 from the contour integral of OUR
#                map (o_scalars/orec.py; the input columns are read only for the frame times); q95_blend where either contour is missing or the
#                ratio is outside [0.5, 2]. No fitted constant: q95 follows the corrector's map change.
#   B7w5 (beta_N): the blend's beta_N with a centred 5-frame median instead of 3 (q95 keeps 3); same validity window, in-shot
#                interpolation and clip as the packer (o_scalars/oapply.py --prepack-q 3 --prepack-b 5, then the packer with --smooth 1).
# Both rules were accepted on DIII-D (holdout + untouched extra set, all frames) on this base (METHODS.md section 3).
SCAL_DIR=$WS/stage11_run/o_scalars
POST=("${POST[@]}" a1w5)

post_a1w5() {  # <run dir> <input npz = the corrected file> <output npz> <mast shot dir>
  local O=$1 IN=$2 OUTF=$3 MD=$4 X=$1/scal C=$1/corr/out D=$1/$CORR_IN_SUB W=$GEOM/mast_limiter_mask.npz
  [ -s "$C/files_corr_mast.txt" ] || { echo "a1w5: no corrected parts in $C (post stage corr first)"; return 1; }
  [ -s "$D/files_${CORR_IN_TAG}_mast.txt" ] || { echo "a1w5: no base parts in $D"; return 1; }
  rm -rf "$X"; mkdir -p "$X/rec" "$X/out"
  ( cd "$SCAL_DIR" && $PY orec.py --machine mast --run "$C" --tag corr --data-dir "$MD" --wall "$W" --out "$X/rec/final.parquet" --procs "$NPROC"
    $PY orec.py --machine mast --run "$D" --tag "$CORR_IN_TAG" --data-dir "$MD" --wall "$W" --out "$X/rec/base.parquet" --procs "$NPROC" ) 2>&1 \
      | grep -E "OREC|Error|Traceback" | cut -c1-400
  [ -s "$X/rec/final.parquet" ] && [ -s "$X/rec/base.parquet" ] || { echo "a1w5: no record tables"; return 1; }
  ( cd "$SCAL_DIR" && $PYS oapply.py --machine mast --run "$C" --tag corr --out "$X/out" --newtag a1w5 --q95 diff --betaN keep \
      --rec-final "$X/rec/final.parquet" --rec-base "$X/rec/base.parquet" --prepack-q 3 --prepack-b 5 ) 2>&1 | grep -E "OAPPLY|Error|Traceback|assert" | cut -c1-600
  [ -s "$X/out/calib_a1w5.json" ] || { echo "a1w5: apply failed"; return 1; }
  ( cd "$SYN_DIR" && $PYS syn_pack.py --parts "$X/out:a1w5" --q95 "net:$X/out:a1w5:$X/out/calib_a1w5.json" --betaN "net:$X/out:a1w5:$X/out/calib_a1w5.json" \
      --base-sub "$IN" --name a1w5 --out-dir "$X" --smooth 1 ) 2>&1 | grep -E "PACK|Error|Traceback|assert" | cut -c1-400
  [ -s "$X/submission/a1w5/mast_public_test.npz" ] && cp "$X/submission/a1w5/mast_public_test.npz" "$OUTF"
}
