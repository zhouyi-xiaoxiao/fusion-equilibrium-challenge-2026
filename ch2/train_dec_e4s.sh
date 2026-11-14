#!/bin/bash
# Derivation of the decoder constants of the profile dec_e4s_c1nx (ch2/pipelines/dec_e4s.inc):
#   ch2/constants/dec_e4s/wD_L.json                     DIII-D source weights, flux weight, prior widths of the decoder coordinates
#   ch2_models/dec_e4s/support_source_d3d.parquet       support-gate source table (weights/MANIFEST_CH2.sha256)
#   ch2/constants/dec_e4s/wS_L.json                     synthetic-MAST weights (from the four population tables, see stage ws)
# DIII-D TRAIN labels only for wD_L.json and the support table: they are computed on the training-pool runs of ch2/train_corr_b50.sh
# (same run name; its stages lists inv_* net_* tab_* must have run: inverse, network and blend parts of the 4,104 pool shots, 12 frames
# each, clean / stressed). No MAST file is read by this script.
#
#   bash ch2/train_dec_e4s.sh <diii_d_train_dir> <run_name of ch2/train_corr_b50.sh> [stage ...]
#   stages (default all, in this order): tab_clean tab_stress est fit support ws compare
#   tab_*    published functionals (public wall and the scorer's envelope extraction) of the pool's inverse maps and network maps, and of
#            the LABEL maps (k_corrector/ktab.py, as stage tab_* of train_corr_b50.sh does for the blend maps)
#   est      estimator tables of the pool (o_decoder/dw.py est: functionals of the inverse map I, of the network map N, of the blend B, the
#            network's auxiliary heads, the frame regime) -> est/L_trc.parquet, est/L_trs.parquet
#   fit      per functional and regime, convex weights of I and N from uncentred out-of-fold error moments against the label functionals;
#            flux weight; prior widths of the decoder's response coordinates (robust sd of the blend's control-point errors / minor radius)
#            -> wD_L.json
#   support  source table of the support gate: the 11 dsup.py feature columns of the two estimator tables, in row order -> support_source_d3d.parquet
#   ws       robust synthetic-MAST weights (o_decoder/dsyn.py ws: w_S = argmin_w max over the populations of the normalised squared error of
#            w N + (1 - w) I, regulariser 0.05 toward 1/2, feasible interval at tolerance 0.02, l_i fixed at 1/2) from the population tables
#            S_sA1, S_sA2, S_sC1, S_sC2 in WS_POP_DIR (default $SCR/ch2_models/dec_e4s/pop; weights/DERIVATION_CH2.sha256). Those tables hold the
#            published functionals of OUR OWN synthetic free-boundary equilibria (driven by the input columns of every 3rd MAST public-test shot)
#            and of the inverse / network / blend maps reconstructed from their rendered diagnostics; they are inputs of this stage, not
#            regenerated here.
#   compare  COMPARE lines of every regenerated file against the shipped one
# Output: $SCR/ch2_corr_d3d/<run_name>/dec/
set -uo pipefail
. "$(dirname "$0")/../env.sh"; export PIPE=syn_xa_p7li50; . "$FEC_REPO/ch2/common.sh"
DD=$(readlink -f "${1:?usage: train_dec_e4s.sh <diii_d_train_dir> <run_name of train_corr_b50.sh> [stages]}"); NAME=${2:?run name}; shift 2
STAGES=${*:-tab_clean tab_stress est fit support ws compare}
R=$SCR/ch2_corr_d3d/$NAME; O=$R/dec; mkdir -p "$O/tab" "$O/est" "$O/logs"; T0=$(date +%s)
KC=$WS/stage11_run/k_corrector; DEC=$WS/stage11_run/o_decoder; KPY=$WS/.vendor_scorer
SHIP=$FEC_REPO/ch2/constants/dec_e4s; SHIP_SUP=${DEC_SUPSRC:-$SCR/ch2_models/dec_e4s/support_source_d3d.parquet}
WS_POP_DIR=${WS_POP_DIR:-$SCR/ch2_models/dec_e4s/pop}
cond() { if [ "$1" = clean ]; then S=trc; else S=trs; fi; }
echo "=== ch2 train_dec_e4s $(hostname) $(date -u) train_dir=$DD pool_run=$R stages=$STAGES"
for st in $STAGES; do case $st in
tab_clean|tab_stress)
  cond "${st#tab_}"
  [ -s "$R/gen/files_li_${S}_d3d.txt" ] && [ -s "$R/pred/files_Cxa32fng1_${S}_d3d.txt" ] || { echo "run ch2/train_corr_b50.sh $NAME (inv_* net_*) first"; exit 1; }
  cp -n "$R/gen/frames_li_${S}_d3d.parquet" "$R/pred/frames_Cxa32fng1_${S}_d3d.parquet"
  for RT in "$R/gen:li_$S" "$R/pred:Cxa32fng1_$S"; do
    ( cd "$KC" && PYTHONPATH=$KPY $PYS ktab.py --machine d3d --run "${RT%%:*}" --tag "${RT##*:}" --wall "$GEOM/d3d_limiter_mask.npz" --data-dir "$DD" \
        --out "$O/tab/${RT##*:}.parquet" --procs "$NPROC" ) 2>&1 | grep -E "KTAB|Error|Traceback" | cut -c1-400
  done
  [ -s "$R/tab/bl50_$S.parquet" ] || { echo "no blend table $R/tab/bl50_$S.parquet (train_corr_b50.sh tab_${st#tab_})"; exit 1; } ;;
est)
  for S in trc trs; do
    ( cd "$DEC" && $PYS dw.py est --machine d3d --inv-tab "$O/tab/li_$S.parquet" --net-tab "$O/tab/Cxa32fng1_$S.parquet" --blend-tab "$R/tab/bl50_$S.parquet" \
        --net-run "$R/pred" --net-tag "Cxa32fng1_$S" --out "$O/est/L_$S.parquet" ) 2>&1 | grep -E "DWEST|Error|Traceback" | cut -c1-400
    [ -s "$O/est/L_$S.parquet" ] || { echo "EST_FAILED $S"; exit 1; }
  done ;;
fit)
  ( cd "$DEC" && $PYS dw.py fit --est "$O/est/L_trc.parquet" "$O/est/L_trs.parquet" --out "$O/wD_L.json" ) > "$O/logs/fit.log" 2>&1
  grep -E "DWFIT|Error|Traceback" "$O/logs/fit.log" | cut -c1-400; [ -s "$O/wD_L.json" ] || { echo FIT_FAILED; exit 1; } ;;
support)
  $PYS - "$O/est/L_trc.parquet" "$O/est/L_trs.parquet" "$O/support_source_d3d.parquet" <<'PY'
import sys
import pandas as pd
C = ["file", "k", "ok"] + [f"{s}_{c}" for c in ("Ra", "Za", "Ri", "Ro", "kappa", "tri_top", "tri_bot", "volume", "li") for s in ("I", "N")]
d = pd.concat([pd.read_parquet(f)[C] for f in sys.argv[1:3]], ignore_index=True)
d.to_parquet(sys.argv[3])
print(f"SUPPORT source table {sys.argv[3]}: {len(d)} rows, {d['file'].nunique()} shots, {len(C)} columns")
PY
  ;;
ws)
  P=""; for s in sA1 sA2 sC1 sC2; do [ -s "$WS_POP_DIR/S_$s.parquet" ] || { echo "missing $WS_POP_DIR/S_$s.parquet"; exit 1; }; P="$P $s=$WS_POP_DIR/S_$s.parquet"; done
  ( cd "$DEC" && $PYS dsyn.py ws --pop $P --neutral li --out "$O/wS_L.json" ) > "$O/logs/ws.log" 2>&1
  [ -s "$O/wS_L.json" ] || { echo WS_FAILED; tail -3 "$O/logs/ws.log"; exit 1; } ;;
compare)
  $PYS - "$O" "$SHIP" "$SHIP_SUP" <<'PY'
import json, math, sys
from pathlib import Path
import pandas as pd
o, ship, sup = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def walk(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append((path + "/" + str(k), "missing")); continue
            walk(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))): out.append((path, abs(a - b)))
    elif a != b: out.append((path, "differs"))
    return out
for n in ("wD_L.json", "wS_L.json"):
    if not (o / n).exists(): print(f"COMPARE {n}: not regenerated"); continue
    d = walk(json.load(open(o / n)), json.load(open(ship / n)))
    num = [x for _, x in d if isinstance(x, float)]
    print(f"COMPARE {n}: {'IDENTICAL' if not d else f'{len(d)} fields differ (max |numeric diff| {max(num) if num else 0:.3e})'}")
    for p, x in d[:10]: print("  differs:", p, x)
if (o / "support_source_d3d.parquet").exists() and sup.exists():
    a, b = pd.read_parquet(o / "support_source_d3d.parquet"), pd.read_parquet(sup)
    same = list(a.columns) == list(b.columns) and len(a) == len(b) and all(a[c].equals(b[c]) for c in a.columns)
    print(f"COMPARE support_source_d3d.parquet (values, row order): {'IDENTICAL' if same else 'DIFFERENT'}")
PY
  ;;
*) echo "unknown stage $st"; exit 2;; esac; echo "--- [$(( $(date +%s) - T0 ))s] $st"; done
echo "CH2_DEC_D3D_DONE $O"
