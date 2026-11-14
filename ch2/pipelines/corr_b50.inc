# Post stage "corr" shared by the corrector profiles (int_b50c1nx, int_b50gld, dec_e4s_c1nx). Not a profile: it is sourced by a profile
# AFTER syn_xa_p7li50.sh (whose post stage "synblend" writes the blend parts this stage reads), and needs CORR_FLAGS (the setting of the
# corrector, see below). The corrected base is the blend (parts <run dir>/syn/blend, tag mbl) unless the profile sets CORR_IN_SUB /
# CORR_IN_TAG before sourcing this file (dec_e4s_c1nx: the decoded parts <run dir>/dec/run, tag dec, written by the post stage "dec").
#
# What it does (label-free on MAST; METHODS.md section 3, PROVENANCE.md section 2):
#   1. table   for EVERY frame of the base map (blend, or decoded blend): magnetic axis and last closed flux surface extracted with the scorer's published
#              functionals and the PUBLIC first-wall polygon (never the envelope), control points of that contour (top, bottom,
#              inboard, outboard), shape scalars, l_i; merged with the per-frame diagnostics of the base inverse run (its frame table).
#   2. apply   41 dimensionless features per frame (shot medians centred on the medians of THIS corpus of our own maps, shot-relative
#              deviations, inverse diagnostics, timing) -> the DIII-D-trained gradient-boosting corrector predicts displacements of the
#              control points (in units of the minor radius) and a log l_i ratio; support gate (1 inside the DIII-D 99 % feature
#              ellipsoid, 0 beyond twice the 99.9 % radius^2); --debias removes the mean of the predicted corrections over THIS corpus
#              (the machine mean of our maps is not moved); per-output strengths; the map is warped by a normalised Gaussian radial
#              basis field through the displaced control points (taper: vacuum region untouched) and, with --li, the interior flux is
#              re-mapped to the corrected l_i.  The corpus statistics (feature centring, zero mean) are recomputed on whatever shots
#              are predicted - on the Final inputs for a Final run.
#   3. pack    psi = corrected maps; q95 / beta_N = the blend's calibrated scalars, unchanged (unit constants); same packer as the blend.
# Model: DIII-D TRAIN labels only (4,104 shots outside every holdout / evaluation / l_i-model list; ch2/train_corr_b50.sh), shipped in
# weights/MANIFEST_CH2.sha256. Contour extraction runs with scikit-image 0.25.2 (the version the organisers' scorer pins), installed by
# env/build_env.sh into src/.vendor_scorer and put in front of the ML environment for this stage only.
# CORR_SHOTS=<file of shot file names> (audit / rehearsal option): the table and its corpus statistics still cover every shot, but only
# the listed shots are corrected; the others keep the blend map (the output is then NOT the profile's file).
CORR_DIR=$WS/stage11_run/k_corrector
CORR_MODEL=${CORR_MODEL:-$SCR/ch2_models/corr_b50/mB50_tr.pkl}
CORR_PYPATH=$WS/.vendor_scorer
CORR_IN_SUB=${CORR_IN_SUB:-syn/blend}; CORR_IN_TAG=${CORR_IN_TAG:-mbl}
MODELS+=("$CORR_MODEL")
POST=("${POST[@]}" corr)

post_corr() {  # <run dir> <input npz = the packed base file> <output npz> <mast shot dir>
  local O=$1 IN=$2 OUTF=$3 B=$1/$CORR_IN_SUB T=$CORR_IN_TAG C=$1/corr R n
  [ -d "$CORR_PYPATH/skimage" ] || { echo "corr: $CORR_PYPATH is missing (bash env/build_env.sh)"; return 1; }
  [ -s "$B/files_${T}_mast.txt" ] || { echo "corr: no base parts in $B (run the post stage that writes them first)"; return 1; }
  [ -s "$O/frames_${TAG}_mast.parquet" ] || { echo "corr: no frame table of the base inverse run ($O/frames_${TAG}_mast.parquet)"; return 1; }
  rm -rf "$C"; mkdir -p "$C/view" "$C/out/parts"
  ln -s "$B/parts" "$C/view/parts"; cp "$B/files_${T}_mast.txt" "$C/view/"; cp "$O/frames_${TAG}_mast.parquet" "$C/view/frames_${T}_mast.parquet"
  ( cd "$CORR_DIR" && PYTHONPATH=$CORR_PYPATH $PYS ktab.py --machine mast --run "$C/view" --tag "$T" --wall "$GEOM/mast_limiter_mask.npz" \
      --out "$C/tab_$T.parquet" --procs "$NPROC" ) 2>&1 | grep -E "KTAB|Error|Traceback" | cut -c1-400
  [ -s "$C/tab_$T.parquet" ] || { echo "corr: no table"; return 1; }
  R=$C/view
  if [ -n "${CORR_SHOTS:-}" ]; then                  # corrected subset only (audit option); corpus statistics from the full table
    R=$C/sel; mkdir -p "$R/parts"; cp "$B/files_${T}_mast.txt" "$R/"
    $PYS - "$B/files_${T}_mast.txt" "$CORR_SHOTS" "$B/parts" "$R/parts" "$T" <<'PY'
import os, sys
names = [x for x in open(sys.argv[1]).read().split("\n") if x]; want = {x for x in open(sys.argv[2]).read().split("\n") if x}
t = sys.argv[5]; n = 0
for i, f in enumerate(names):
    if f in want:
        os.symlink(f"{sys.argv[3]}/{t}_mast_{i:05d}.npz", f"{sys.argv[4]}/{t}_mast_{i:05d}.npz"); n += 1
print(f"corr: CORR_SHOTS selects {n} of {len(names)} shots ({len(want - set(names))} listed names not in this run)")
PY
  fi
  ( cd "$CORR_DIR" && PYTHONPATH=$CORR_PYPATH $PYS kapply.py --machine mast --run "$R" --tag "$T" --table "$C/tab_$T.parquet" --model "$CORR_MODEL" \
      --out "$C/out" --newtag corr --wall "$GEOM/mast_limiter_mask.npz" "${CORR_FLAGS[@]}" --procs "$NPROC" ) 2>&1 | grep -E "KAPPLY|Error|Traceback" | cut -c1-900
  [ -s "$C/out/apply_corr_mast.json" ] || { echo "corr: apply failed"; return 1; }
  if [ -n "${CORR_SHOTS:-}" ]; then                  # shots not selected: the base part unchanged
    n=$(grep -c . "$B/files_${T}_mast.txt")
    for i in $(seq 0 $((n - 1))); do f=$(printf "%05d" "$i"); [ -e "$C/out/parts/corr_mast_$f.npz" ] || ln -s "$B/parts/${T}_mast_$f.npz" "$C/out/parts/corr_mast_$f.npz"; done
  fi
  echo '{"C_q95": 1.0, "C_betaN": 1.0, "source": "unit: the blend parts carry the calibrated scalars"}' > "$C/calib_unit.json"
  ( cd "$SYN_DIR" && $PYS syn_pack.py --parts "$C/out:corr" --q95 "net:$C/out:corr:$C/calib_unit.json" --betaN "net:$C/out:corr:$C/calib_unit.json" \
      --base-sub "$IN" --name corr --out-dir "$C" ) 2>&1 | grep -E "PACK|Error|Traceback|assert" | cut -c1-400
  [ -s "$C/submission/corr/mast_public_test.npz" ] && cp "$C/submission/corr/mast_public_test.npz" "$OUTF"
}
