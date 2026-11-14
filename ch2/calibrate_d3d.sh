#!/bin/bash
# Ch2 "training" entrypoint: regenerates the DIII-D-derived constants of the selected MAST pipeline. CPU only. DIII-D labels only;
# no MAST file is read.
#
#   [PIPE=<profile>] bash ch2/calibrate_d3d.sh <diii_d_train_dir> <run_name> [stage ...]      stages: pcal (default) | labelfit
#
#   pcal      the two constants of ch2/constants/<PIPE>: C_q95 (calib_<tag>.json) and c_betaN (c_<tag>_mast.json). The pipeline's own
#             solver is run on DIII-D TRAIN shots (holdout excluded, every 23rd shot, 8 frames each = 300 shots) with DIII-D's Thomson
#             data re-sampled on a MAST-like horizontal midplane chord ("pseudo-horizontal"); C_q95 = median(label q95 / own q95),
#             c_betaN = least squares through the origin of label beta_N on the electron beta_N,e of core-covered frames.
#   labelfit  the statistics behind the prior means / widths of the profile family (only for profiles that define labelfit()).
#   <other>   a profile-defined stage calib_<other>(), e.g. "lifit" of p7li (the l_i-from-ramp-history model, DIII-D labels only).
# Output: $SCR/ch2_calib/<run_name>/ ; compare with ch2/constants/<PIPE> via the printed COMPARE lines.
set -uo pipefail
. "$(dirname "$0")/../env.sh"; . "$FEC_REPO/ch2/common.sh"
DD=$(readlink -f "${1:?usage: calibrate_d3d.sh <diii_d_train_dir> <run_name> [stages]}"); NAME=${2:?run name}; shift 2; STAGES=${*:-pcal}
O=$SCR/ch2_calib/$NAME; mkdir -p "$O"; T0=$(date +%s)
names_dir "$FEC_REPO/splits/d3d_subset1150.txt" "$SCR/splits/d3d_subset1150"; HOLD=$SCR/splits/d3d_subset1150:1000:1150
echo "=== ch2 calibrate $(hostname) $(date -u) train_dir=$DD ($(ls "$DD" | grep -c parquet) files) pipeline=$PIPE tag=$TAG cores=$NPROC stages=$STAGES"
for st in $STAGES; do case $st in
pcal)
  run_solver --machine d3d --mode pseudo --data-dir "$DD" --exclude-holdout "$HOLD" --stride 23 --wall "$GEOM/d3d_limiter_mask.npz" \
      "${PCAL_ARGS[@]}" --tag "${TAG}_pcal" --out "$O" --frames-per-shot 8 --procs "$NPROC" 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-800
  $PYS "$G1/eval_frames.py" --run "$O" --tag "${TAG}_pcal" --data-dir "$DD" --self-calib --procs 48 2>&1 | grep -E "EVAL|gauge|Error|Traceback" | cut -c1-800
  [ -s "$O/calib_${TAG}_pcal.json" ] || { echo "PCAL_FAILED: no calib_${TAG}_pcal.json in $O"; exit 1; }
  cp "$O/calib_${TAG}_pcal.json" "$O/calib_${TAG}.json"
  $PY "$SCALC/scal_frames.py" --machine d3d --src g1 --run "$O" --tag "${TAG}_pcal" --data-dir "$DD" --wall "$GEOM/d3d_limiter_mask.npz" --out "$O/scal_d3d_${TAG}_pcal.parquet" --procs "$NPROC" | grep SUMMARY | cut -c1-400
  $PYS "$INV6/scal_c.py" "$O/scal_d3d_${TAG}_pcal.parquet" "$O/c_${TAG}_mast.json" --use core | cut -c1-600
  [ -s "$O/c_${TAG}_mast.json" ] || { echo "PCAL_FAILED: no c_${TAG}_mast.json in $O"; exit 1; }
  $PYS - "$O" "$CONST" "$TAG" <<'PY'
import json, sys
o, c, t = sys.argv[1:4]
a = json.load(open(f"{o}/calib_{t}.json"))["C_q95"]; b = json.load(open(f"{c}/calib_{t}.json"))["C_q95"]
x = json.load(open(f"{o}/c_{t}_mast.json"))["c_used"]; y = json.load(open(f"{c}/c_{t}_mast.json"))["c_used"]
print(f"COMPARE C_q95   new {a!r} shipped {b!r} {'IDENTICAL' if a == b else 'rel diff %.3e' % (a / b - 1)}")
print(f"COMPARE c_betaN new {x!r} shipped {y!r} {'IDENTICAL' if x == y else 'rel diff %.3e' % (x / y - 1)}")
PY
  ;;
labelfit)
  if declare -F labelfit > /dev/null; then labelfit "$DD" "$O" "$HOLD" || echo "LABELFIT returned a non-zero status (check the lines above)"; else echo "pipeline $PIPE defines no labelfit stage (see its profile)"; fi
  ;;
*)
  # further profile-defined DIII-D stages: function calib_<stage> <diii_d_train dir> <out dir> <holdout spec> (e.g. lifit of p7li)
  if declare -F "calib_$st" > /dev/null; then "calib_$st" "$DD" "$O" "$HOLD" || { echo "STAGE_FAILED $st"; exit 1; }; else echo "unknown stage $st (pipeline $PIPE)"; exit 2; fi
  ;; esac; echo "--- [$(( $(date +%s) - T0 ))s] $st"; done
echo "CH2_CALIB_DONE $O"
