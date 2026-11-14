# Post stage "dec" (start-up decoder) of the profile dec_e4s_c1nx. Not a profile: it is sourced by a profile AFTER syn_xa_p7li50.sh
# (whose post stage "synblend" writes the 0.5 / 0.5 blend parts this stage starts from) and BEFORE corr_b50.inc (whose post stage "corr"
# then reads the decoded parts: CORR_IN_SUB / CORR_IN_TAG are set here).
#
# What it does (label-free on MAST; METHODS.md section 3, PROVENANCE.md section 2). Every step runs on ALL frames of the shots being
# predicted, and every statistic over frames or shots is computed here from this run's own maps and inputs:
#   1. tables  published functionals (the scorer's code, PUBLIC first-wall polygon, never the envelope) of the three maps of every frame:
#              the inverse (INV_P7li), the network and the blend; merged with the inverse's per-frame diagnostics (its frame table).
#   2. est     one estimator table: functionals of the inverse map (I) and of the network map (N), the network's auxiliary heads, and the
#              frame regime: start-up = the inverse's contour is not X-point-bounded, or I_p < 0.6 of the shot maximum within the first
#              35 % of the shot; inner-censored; near-double-null; ordinary (same a-priori rule on both machines).
#   3. sup     support gate of the DIII-D source weights: whitened 32-nearest-neighbour distance of each frame's 11 dimensionless
#              net-minus-inverse features to the DIII-D training pool (support source table, shipped: weights/MANIFEST_CH2.sha256);
#              whitening and the leave-one-shot-out thresholds come from that DIII-D table only.
#   4. apply   per-frame targets of the 7 scorer functionals and 6 boundary control coordinates: convex combination of I and N with
#              weight (1 - g) w_S + g w_D, g = 0.5 x support, projected on the feasible interval of w_S; w_D = DIII-D TRAIN-pool source
#              weights (ch2/constants/dec_e4s/wD_L.json), w_S = robust weights on our own synthetic MAST-geometry equilibria
#              (ch2/constants/dec_e4s/wS_L.json); target scales s_j = standard deviation of THIS run's targets over its frames.
#   5. decode  the frozen decoder (ch2/constants/dec_e4s/decoder_E4.json) on the start-up frames only: 12 smooth, current-neutral
#              response coordinates of the blend's plasma flux, 3 Gauss-Newton steps toward the targets (prior widths and trust region
#              from DIII-D training errors, wD_L.json "prior"); pooled flux variance v_psi = variance of THIS run's blend maps over all
#              frames; every other frame keeps the blend map bit for bit; q95 / beta_N = the blend's.
#   6. pack    the decoded maps with the blend's scalars (unit constants), same packer as the blend (the post stage's output file).
# The corpus statistics of steps 4 and 5 (target scales, v_psi), the per-frame support distances and, in the following "corr" stage,
# the corrector's feature centring and zero mean are therefore recomputed on the Final inputs for a Final run.
# Contour extraction runs with scikit-image 0.25.2 (the version the organisers' scorer pins), src/.vendor_scorer (env/build_env.sh).
DEC_DIR=$WS/stage11_run/o_decoder
DEC_KC=$WS/stage11_run/k_corrector
DEC_PYPATH=$WS/.vendor_scorer
DEC_CONST=$FEC_REPO/ch2/constants/dec_e4s
DEC_CFG=${DEC_CFG:-$DEC_CONST/decoder_E4.json}
DEC_WD=$DEC_CONST/wD_L.json
DEC_WS=$DEC_CONST/wS_L.json
DEC_GD=0.5
DEC_SUPSRC=${DEC_SUPSRC:-$SCR/ch2_models/dec_e4s/support_source_d3d.parquet}
EXTRA_CONST=${EXTRA_CONST:+$EXTRA_CONST:}$DEC_CONST
MODELS+=("$DEC_SUPSRC")
POST=("${POST[@]}" dec)
CORR_IN_SUB=dec/run; CORR_IN_TAG=dec

post_dec() {  # <run dir> <input npz = the packed blend file> <output npz> <mast shot dir>
  local O=$1 IN=$2 OUTF=$3 B=$1/syn/blend P=$1/syn/pred R=$1/syn/real X=$1/dec W=$GEOM/mast_limiter_mask.npz NT=${SYN_NAME}fg1 f
  [ -d "$DEC_PYPATH/skimage" ] || { echo "dec: $DEC_PYPATH is missing (bash env/build_env.sh)"; return 1; }
  [ -s "$B/files_mbl_mast.txt" ] || { echo "dec: no blend parts in $B (post stage synblend first)"; return 1; }
  [ -s "$P/files_${NT}_mast.txt" ] || { echo "dec: no network prediction in $P (stage syntrain)"; return 1; }
  [ -s "$R/files_mast_all.txt" ] || { echo "dec: no network input features in $R (stage synreal)"; return 1; }
  [ -s "$O/frames_${TAG}_mast.parquet" ] || { echo "dec: no frame table of the base inverse run ($O/frames_${TAG}_mast.parquet)"; return 1; }
  for f in "$DEC_CFG" "$DEC_WD" "$DEC_WS" "$DEC_SUPSRC"; do [ -s "$f" ] || { echo "dec: missing $f"; return 1; }; done
  rm -rf "$X"; mkdir -p "$X/view_N" "$X/view_B" "$X/tab" "$X/sup" "$X/run"
  ln -s "$P/parts" "$X/view_N/parts"; cp "$P/files_${NT}_mast.txt" "$X/view_N/"; cp "$O/frames_${TAG}_mast.parquet" "$X/view_N/frames_${NT}_mast.parquet"
  ln -s "$B/parts" "$X/view_B/parts"; cp "$B/files_mbl_mast.txt" "$X/view_B/"; cp "$O/frames_${TAG}_mast.parquet" "$X/view_B/frames_mbl_mast.parquet"
  ( cd "$DEC_KC" && export PYTHONPATH=$DEC_PYPATH
    $PYS ktab.py --machine mast --run "$O" --tag "$TAG" --wall "$W" --out "$X/tab/I.parquet" --procs "$NPROC"
    $PYS ktab.py --machine mast --run "$X/view_N" --tag "$NT" --wall "$W" --out "$X/tab/N.parquet" --procs "$NPROC"
    $PYS ktab.py --machine mast --run "$X/view_B" --tag mbl --wall "$W" --out "$X/tab/B.parquet" --procs "$NPROC" ) 2>&1 | grep -E "KTAB|Error|Traceback" | cut -c1-300
  for f in I N B; do [ -s "$X/tab/$f.parquet" ] || { echo "dec: no table $f"; return 1; }; done
  ( cd "$DEC_DIR" && $PYS dw.py est --machine mast --inv-tab "$X/tab/I.parquet" --net-tab "$X/tab/N.parquet" --blend-tab "$X/tab/B.parquet" \
      --net-run "$P" --net-tag "$NT" --out "$X/est_m.parquet" ) 2>&1 | grep -E "DWEST|Error|Traceback" | cut -c1-400
  [ -s "$X/est_m.parquet" ] || { echo "dec: no estimator table"; return 1; }
  ( cd "$DEC_DIR" && $PYS dsup.py --source "$DEC_SUPSRC" --query "m=$X/est_m.parquet" --out "$X/sup" ) 2>&1 | grep -E "DSUP|Error|Traceback" | cut -c1-400
  [ -s "$X/sup/sup_m.parquet" ] || { echo "dec: no support table"; return 1; }
  ( cd "$DEC_DIR" && $PYS dw.py apply --est "$X/est_m.parquet" --weights "$DEC_WD" --set IN --gd "$DEC_GD" --ws "$DEC_WS" \
      --support "$X/sup/sup_m.parquet" --out "$X/tg_m.parquet" ) 2>&1 | grep -E "DWAPPLY|Error|Traceback" | cut -c1-600
  [ -s "$X/tg_m.parquet" ] && [ -s "$X/tg_m_stats.json" ] || { echo "dec: no targets"; return 1; }
  ( cd "$DEC_DIR" && PYTHONPATH=$DEC_PYPATH $PYS dec_run.py --machine mast --wall "$W" --inv "$O:$TAG" --net "$P:$NT" --blend "$B:mbl" \
      --real "$R/mast_all" --targets "$X/tg_m.parquet" --prior "$DEC_WD" --cfg "$DEC_CFG" --tag dec --out "$X/run" --procs "$NPROC" ) 2>&1 \
      | grep -E "DECRUN|Error|Traceback|error" | cut -c1-1200
  [ "$(ls "$X/run/parts" 2>/dev/null | grep -c '^dec_mast_.*\.npz$')" = "$(grep -c . "$B/files_mbl_mast.txt")" ] || { echo "dec: decoded parts incomplete"; return 1; }
  echo '{"C_q95": 1.0, "C_betaN": 1.0, "source": "unit: the decoded parts carry the blend calibrated scalars"}' > "$X/calib_unit.json"
  ( cd "$SYN_DIR" && $PYS syn_pack.py --parts "$X/run:dec" --q95 "net:$X/run:dec:$X/calib_unit.json" --betaN "net:$X/run:dec:$X/calib_unit.json" \
      --base-sub "$IN" --name dec --out-dir "$X" ) 2>&1 | grep -E "PACK|Error|Traceback|assert" | cut -c1-400
  [ -s "$X/submission/dec/mast_public_test.npz" ] && cp "$X/submission/dec/mast_public_test.npz" "$OUTF"
}
