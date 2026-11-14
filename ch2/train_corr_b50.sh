#!/bin/bash
# DIII-D derivation of the map corrector of the profiles int_b50c1nx / int_b50gld (ch2/pipelines/corr_b50.inc): regenerates the model
# weights/MANIFEST_CH2.sha256 -> ch2_models/corr_b50/mB50_tr.pkl from DIII-D TRAIN labels only. No MAST file is read.
#
#   [NSHARD=N SHARD=k] bash ch2/train_corr_b50.sh <diii_d_train_dir> <run_name> <twin network .pt> [stage ...]
#   stages (default all, in this order): lists inv_clean inv_stress net_clean net_stress tab_clean tab_stress fit compare
#
#   lists       training pool = DIII-D train shots minus the 150 holdout shots, minus the untouched 150-shot extra set
#               (splits/d3d_extra150.txt), minus the 2,637 training shots of the l_i model (splits/d3d_li_train.txt; the l_i datum of the
#               inverse would be in-sample there) = 4,104 shots; checked against splits/d3d_corr_pool.txt
#   inv_*       the p7li inverse (profile p7li: same solver, configuration, l_i model, fallback gate) on 12 evenly spaced frames of every pool
#               shot, MAST-like pseudo-horizontal Thomson chord, clean / stressed (noise 10 %, branch 5 %, timing 10 ms, separatrix 3 cm,
#               edge noise 1 cm). NSHARD / SHARD split the shots into independent jobs; a call without SHARD collects (and solves any
#               shot that is missing). About 7,000 allocated cpu-minutes per condition (4 x 144 cores, 12 min).
#   net_*       the DIII-D twin network (third argument: model_Cxa32.pt written by ch2/train_syn_d3d.sh; the network of the syn_* profiles
#               trained on DIII-D-driven synthetic equilibria) on the same shots: network inputs of all frames (same pseudo-noise stream as
#               the inverse), prediction with the production post-processing (fill, temporal sigma 1 frame), the inverse's 12 frames kept
#   tab_*       blend 0.5 x network + 0.5 x inverse (each with its own DIII-D calibration constants, as in syn_xa_p7li50), then the
#               corrector table: geometry of OUR map (public-wall extraction, and the scorer's envelope extraction) and of the LABEL map
#               (scorer's envelope extraction), per frame
#   fit         gradient boosting (LightGBM, hyper-parameters fixed a priori), 9 outputs (8 control-point displacements / minor radius,
#               log l_i ratio), 41 dimensionless features (k_corrector/kcommon.py, set "tr"), shot-grouped 5-fold CV report,
#               leave-configuration-out report, support statistics for the gate -> model/mB50_tr.pkl
#   compare     predictions of the regenerated and of the shipped model on the regenerated tables (COMPARE line). LightGBM's multi-threaded
#               histogram sums are not bit-reproducible: in development a refit from bit-identical tables gave 3 of 9 boosters text-identical
#               and predictions within 1.1e-16 (no effect on the corrected maps of a 25-shot MAST stride sample)
# Output: $SCR/ch2_corr_d3d/<run_name>/
set -uo pipefail
. "$(dirname "$0")/../env.sh"; export PIPE=syn_xa_p7li50; . "$FEC_REPO/ch2/common.sh"     # solver + configuration of p7li, network inputs of syn_net.inc
DD=$(readlink -f "${1:?usage: train_corr_b50.sh <diii_d_train_dir> <run_name> <twin model .pt> [stages]}"); NAME=${2:?run name}
TWIN=$(readlink -f "${3:?twin network model_Cxa32.pt (ch2/train_syn_d3d.sh)}"); shift 3
STAGES=${*:-lists inv_clean inv_stress net_clean net_stress tab_clean tab_stress fit compare}
O=$SCR/ch2_corr_d3d/$NAME; mkdir -p "$O/logs" "$O/gen" "$O/pred" "$O/tab" "$O/model"; T0=$(date +%s)
names_dir "$FEC_REPO/splits/d3d_subset1150.txt" "$SCR/splits/d3d_subset1150"; HOLD=$SCR/splits/d3d_subset1150:1000:1150
POOL=$FEC_REPO/splits/d3d_corr_pool.txt
STRESS='{"noise":0.1,"branch":0.05,"timing":10.0,"sep":0.03,"edge_noise":0.01}'
SYN_DIR=$WS/stage11_run/j_synth_net; KC=$WS/stage11_run/k_corrector; KPY=$WS/.vendor_scorer
CN=$FEC_REPO/ch2/constants/syn_xa/calib_Cxa32.json; CL=$FEC_REPO/ch2/constants/p7li/calib_C4F.json
SHIPPED=${CORR_MODEL:-$SCR/ch2_models/corr_b50/mB50_tr.pkl}
cond() { if [ "$1" = clean ]; then ST='{}'; S=trc; else ST=$STRESS; S=trs; fi; }
echo "=== ch2 train_corr_b50 $(hostname) $(date -u) train_dir=$DD twin=$TWIN stages=$STAGES nshard=${NSHARD:-1} shard=${SHARD:-all}"
for st in $STAGES; do case $st in
lists)
  $PYS - "$DD" "$FEC_REPO/splits/d3d_subset1150.txt" "$FEC_REPO/splits/d3d_extra150.txt" "$FEC_REPO/splits/d3d_li_train.txt" "$POOL" "$O/pool.txt" <<'PY'
import os, sys
dd, sub, ext, lit, ship, out = sys.argv[1:7]
rd = lambda p: [x for x in open(p).read().split("\n") if x]
d = sorted(x for x in os.listdir(dd) if x.endswith(".parquet"))
hold = set(sorted(rd(sub))[1000:1150]); e = set(rd(ext)); li = set(rd(lit))
pool = [x for x in d if x not in hold and x not in e and x not in li]
open(out, "w").write("\n".join(pool) + "\n")
print(f"LISTS train {len(d)} holdout {len(hold)} extra {len(e)} l_i train {len(li)} -> pool {len(pool)}; "
      f"COMPARE with splits/d3d_corr_pool.txt: {'IDENTICAL' if pool == rd(ship) else 'DIFFERENT'}")
PY
  ;;
inv_clean|inv_stress)
  cond "${st#inv_}"; SA=(); [ -n "${SHARD:-}" ] && SA=(--shard "$SHARD/${NSHARD:?NSHARD}")
  run_solver --machine d3d --mode pseudo --data-dir "$DD" --exclude-holdout "$HOLD" --shot-list "$POOL" --wall "$GEOM/d3d_limiter_mask.npz" \
      --set "stress=$ST" --frames-per-shot 12 --tag "li_$S" --out "$O/gen" "${SA[@]}" --procs "$NPROC" >> "$O/logs/$st${SHARD:+_$SHARD}.log" 2>&1
  grep -E "SUMMARY|shard|Traceback|Error" "$O/logs/$st${SHARD:+_$SHARD}.log" | tail -2 | cut -c1-600
  [ -n "${SHARD:-}" ] || [ -s "$O/gen/files_li_${S}_d3d.txt" ] || { echo "INV_INCOMPLETE $st"; exit 1; } ;;
net_clean|net_stress)
  cond "${st#net_}"
  [ -s "$O/real/files_tr_$S.txt" ] || ( cd "$SYN_DIR" && $PY syn_gen.py --mode real --machine d3d --data-dir "$DD" --exclude-holdout "$HOLD" --shot-list "$POOL" \
      --wall "$GEOM/d3d_limiter_mask.npz" "${SYN_CFG_REAL[@]}" --set "stress=$ST" --tag "tr_$S" --out "$O" --procs "$NPROC" ) > "$O/logs/${st}_real.log" 2>&1
  ( cd "$SYN_DIR" && $PYS syn_predict.py --model "$TWIN" --real "$O/real/tr_$S" --machine d3d --tag "Cxa32fng1_$S" --out "$O/pred" --fill-nots --smooth 1 \
      --smooth-scal --subset-ks 12 ) 2>&1 | grep -E "PREDICT|Error|Traceback" | cut -c1-600
  [ -s "$O/pred/files_Cxa32fng1_${S}_d3d.txt" ] || { echo "NET_FAILED $st"; exit 1; } ;;
tab_clean|tab_stress)
  cond "${st#tab_}"
  [ -s "$O/gen/files_bl50_${S}_d3d.txt" ] || ( cd "$SYN_DIR" && $PYS syn_avg.py --runs "$O/pred:Cxa32fng1_$S:$CN" "$O/gen:li_$S:$CL" --weights 0.5,0.5 \
      --out "$O/gen" --tag "bl50_$S" --machine d3d ) 2>&1 | grep -E "AVG|Error|Traceback|assert" | cut -c1-400
  cp -n "$O/gen/frames_li_${S}_d3d.parquet" "$O/gen/frames_bl50_${S}_d3d.parquet"
  ( cd "$KC" && PYTHONPATH=$KPY $PYS ktab.py --machine d3d --run "$O/gen" --tag "bl50_$S" --wall "$GEOM/d3d_limiter_mask.npz" --data-dir "$DD" \
      --out "$O/tab/bl50_$S.parquet" --procs "$NPROC" ) 2>&1 | grep -E "KTAB|Error|Traceback" | cut -c1-400
  [ -s "$O/tab/bl50_$S.parquet" ] || { echo "TAB_FAILED $st"; exit 1; } ;;
fit)
  ( cd "$KC" && PYTHONPATH=$KPY $PYS kfit.py --train "$O/tab/bl50_trc.parquet" "$O/tab/bl50_trs.parquet" --fset tr --out "$O/model/mB50_tr.pkl" --lco ) \
      > "$O/logs/fit.log" 2>&1
  grep -E "KFIT|TARGETS" "$O/logs/fit.log" | cut -c1-600
  [ -s "$O/model/mB50_tr.pkl" ] || { echo "FIT_FAILED"; exit 1; }
  echo "model $(sha256sum "$O/model/mB50_tr.pkl" | cut -c1-64)" ;;
compare)
  ( cd "$KC" && $PYS - "$O/model/mB50_tr.pkl" "$SHIPPED" "$O/tab/bl50_trc.parquet" "$O/tab/bl50_trs.parquet" <<'PY'
import pickle, sys
import numpy as np, pandas as pd
import kcommon as KC
new, old = (pickle.load(open(p, "rb")) for p in sys.argv[1:3])
print("COMPARE columns", "IDENTICAL" if new["cols"] == old["cols"] else "DIFFERENT", "| DIII-D reference medians",
      "IDENTICAL" if new["ref_d3d"] == old["ref_d3d"] else "DIFFERENT")
txt = sum(new["models"][o].booster_.model_to_string() == old["models"][o].booster_.model_to_string() for o in KC.OUTS)
print(f"COMPARE boosters text-identical {txt} of {len(KC.OUTS)}")
for p in sys.argv[3:5]:
    d = pd.read_parquet(p); F, _ = KC.feats(d, new["ref_d3d"]); X = F[new["cols"]]
    dm = max(float(np.max(np.abs(new["models"][o].predict(X) - old["models"][o].predict(X)))) for o in KC.OUTS)
    print(f"COMPARE predictions on {p.rsplit('/', 1)[-1]} ({len(d)} frames): max |new - shipped| = {dm:.3e}")
PY
  ) ;;
*) echo "unknown stage $st"; exit 2;; esac; echo "--- [$(( $(date +%s) - T0 ))s] $st"; done
echo "CH2_CORR_D3D_DONE $O"
