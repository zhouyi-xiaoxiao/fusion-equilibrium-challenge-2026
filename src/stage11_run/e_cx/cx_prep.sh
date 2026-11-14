#!/bin/bash
# Stage 11E / CX — prep: (1) parity of FEATURE-V2.2 under the full mirror (must equal the v2.1 classes),
# (2) mirror-canonical cfl caches on v2.2: h150 <- feat_v22, fold3/fold2 <- ip_axis fold v2.2 caches (scaler refit on each train split),
# (3) shadow evaluation dirs (links to metric/ref/blocks; corrector = per-file links, every new output tagged CX*; reference stats).
# sbatch -J cx_prep -t 00:40:00 job_cpu.sbatch bash cx_prep.sh
set -uo pipefail
mkdir -p "$CX/logs"
python cx_parity.py > "$CX/logs/parity_v22.log" 2>&1; prc=$?
tail -4 "$CX/logs/parity_v22.log"
[ $prc -eq 0 ] || { echo "PARITY_FAILED rc=$prc"; exit 3; }
declare -A SRC=( [h150]=$S11/feat_v22 [fold3]=$IPX/fold3/feat [fold2]=$IPX/fold2/feat )
for s in h150 fold3 fold2; do
  mkdir -p "$CX/$s"
  [ -e "$CX/$s/feat/test/X.npy" ] || ( SRC=${SRC[$s]} DST=$CX/$s/feat VARIANT=cfl PARITY=$CX/parity_v22.npz python cx_canon.py > "$CX/logs/canon_$s.log" 2>&1 ) &
done
wait
for s in h150 fold3 fold2; do echo "== canon $s"; tail -5 "$CX/logs/canon_$s.log"; done
# ---- shadow dirs
lnk() { ln -sfn "$1" "$2"; }
shadow_cc() { local f b; mkdir -p "$2/logs" "$2/scores"; for f in "$1"/*; do b=$(basename "$f"); [ -d "$f" ] && continue; [ -e "$2/$b" ] || ln -s "$f" "$2/$b"; done; }
H=$CX/h150; mkdir -p "$H"/{preds,scores,logs,integrate,official}
lnk "$S11/metric" "$H/metric"; lnk "$S11/fold2/ref_holdout150" "$H/ref_fold2"
shadow_cc "$S11/corrector" "$H/corrector"
HS=$S11/integrate/hold/stats
for p in "FMTW_qSC:$HS/stats_h150_FMTW" "ig_robmiFMTW:$HS/stats_h150_robmiFMTW" "cc_FMTWqSC:$HS/stats_h150_inc" \
         "E_FMTW_qSC:$E/h150_cfl/stats/stats_h150_E_FMTW" "E_ig_robmi:$E/h150_cfl/stats/stats_h150_E_ig" "E_ig_qSCc:$E/h150_cfl/stats/stats_h150_E_ig_qSCc" \
         "X22_FMTW_qSC:$IPX/h150/scores/stats_X22_FMTW_qSC" "X22_ig_robmi:$IPX/h150/scores/stats_X22_ig_robmi"; do
  IFS=: read a b <<< "$p"; [ -e "$b.npz" ] && lnk "$b.npz" "$H/scores/stats_$a.npz" || echo "MISSING $b.npz"
done
for f in fold3 fold2; do
  F=$CX/$f; O=$S11/$f; mkdir -p "$F"/{preds,scores,logs,integrate,official}
  for x in metric ref_fold2 fold2_block.npy gt_list_fold2.txt fold2_files.txt fold2_info.json train_keep_idx.npy fold2_trainidx.npy; do [ -e "$O/$x" ] && lnk "$O/$x" "$F/$x"; done
  shadow_cc "$O/corrector" "$F/corrector"
  for t in FMTW_qSC FM_qSC TW_qSC cc_FMTWqSC ig_robmiFMTW; do [ -e "$O/scores/stats_$t.npz" ] && lnk "$O/scores/stats_$t.npz" "$F/scores/stats_$t.npz"; done
  for p in "E_FMTW_qSC:$O/scores/stats_${f}_cfl_FMTW_qSC" "E_ig_robmi:$O/scores/stats_ig_e_${f}_cfl_FMTWqSC" "E_ig_qSCc:$O/scores/stats_ig_e_${f}_cfl_qSCc" \
           "X22_FMTW_qSC:$IPX/$f/scores/stats_X22_FMTW_qSC" "X22_ig_robmi:$IPX/$f/scores/stats_X22_ig_robmi"; do
    IFS=: read a b <<< "$p"; [ -e "$b.npz" ] && lnk "$b.npz" "$F/scores/stats_$a.npz" || echo "MISSING $b.npz"
  done
done
ls -l "$CX"/*/scores | head -60
# ---- self-checks on the new caches: row identity with the v2.2 sources, odd-column count, sign column
python - <<'PY'
import os, numpy as np
from pathlib import Path
CX=Path(os.environ["CX"]); S11=Path(os.environ["S11"]); IPX=Path(os.environ["IPX"])
src={"h150":S11/"feat_v22","fold3":IPX/"fold3/feat","fold2":IPX/"fold2/feat"}
for s,d in src.items():
    sc=np.load(CX/s/"feat/scaler_v21.npz"); odd=sc["odd"]; c=int(sc["sign_col"])
    for sp in ("train","holdout","test"):
        a=(CX/s/"feat"/sp/"files.txt").read_text(); b=(d/sp/"files.txt").read_text()
        Xr=np.load(d/sp/"X_raw.npy",mmap_mode="r"); X=np.load(CX/s/"feat"/sp/"X.npy",mmap_mode="r")
        sg=np.asarray(Xr[:,c]); 
        print(f"{s}/{sp}: files identical {a==b}; rows {X.shape[0]}=={Xr.shape[0]}; reversed frames {(sg<0).mean():.4f}; odd cols {int(odd.sum())}; variant {sc['variant']}")
PY
