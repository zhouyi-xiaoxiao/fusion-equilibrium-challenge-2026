#!/bin/bash
# Stage 11F / corr_canon — prep (idempotent): own corrector dirs with per-file links to the UNCHANGED family artefacts of each set
# (gv21 OOF teacher statistics: cov / FD steps / dsc / f11_train; the Jacobian + base derivations of the canonical E_cfl base, which do
# not depend on the heads) and the FULL canonical head-training cache.  Nothing outside $F is written.
#   tags inside $F/<set>/corrector:  Fc  = canonical heads + canonical-refit rob constants (candidate)
#                                    Foc = canonical heads + the OLD (raw-head) frozen rob constants (diagnostic)
set -uo pipefail
lnk() { [ -e "$1" ] || { echo "MISSING $1"; return 1; }; ln -sfn "$1" "$2"; }
rc=0
for s in fold3 fold2; do
  O=$S/$s/corrector; C=$F/$s/corrector; B=e_${s}_cfl_FMTWqSC
  mkdir -p "$C"/{logs,scores,tmp} "$F/$s"/{preds,scores,official,logs}
  for f in cov_oof_gv21.npy steps_gv21.npy dsc_gv21.json f11_train_gv21.npy; do lnk "$O/$f" "$C/$f" || rc=1; done
  for t in Fc Foc; do
    lnk "$O/jac_fold2_$B.npz" "$C/jac_fold2_$t.npz" || rc=1; lnk "$O/f11_fold2_$B.npy" "$C/f11_fold2_$t.npy" || rc=1
  done
  lnk "$S/$s/scores/stats_ig_e_${s}_cfl_qSCc.npz" "$F/$s/scores/stats_E_ig_qSCc.npz" || rc=1
  lnk "$S/$s/scores/stats_ig_e_${s}_cfl_FMTWqSC.npz" "$F/$s/scores/stats_E_ig.npz" || rc=1
  lnk "$S/$s/scores/stats_cc_e_${s}_cfl_FMTWqSC.npz" "$F/$s/scores/stats_E_cc.npz" || rc=1
  lnk "$S/$s/scores/stats_${s}_cfl_FMTW_qSC.npz" "$F/$s/scores/stats_E_FMTW_qSC.npz" || rc=1
  lnk "$S/$s/scores/stats_ig_robmiFMTW.npz" "$F/$s/scores/stats_ig_robmiFMTW.npz" || rc=1
done
O=$S/corrector; C=$F/h150/corrector; B=E_h150cfl_FMTW_qSC
mkdir -p "$C"/{logs,scores,tmp} "$F/h150"/{preds,scores,official,logs}
for f in cov_oof_gv21.npy steps_gv21.npy heads_innov_gv21.json f11_train_gv21.npy f11_holdout_gv21.npy hold_ref.npz; do lnk "$O/$f" "$C/$f" || rc=1; done
for t in Fc Foc; do
  lnk "$O/jac_holdout_$B.npz" "$C/jac_holdout_$t.npz" || rc=1; lnk "$O/f11_holdout_$B.npy" "$C/f11_holdout_$t.npy" || rc=1
  lnk "$O/der_hold_$B.npz" "$C/der_hold_$t.npz" || rc=1
done
for p in "E_ig_qSCc:h150_E_ig_qSCc" "E_ig:h150_E_ig" "E_cc:h150_E_cc" "E_FMTW_qSC:h150_E_FMTW" "ig_robmiFMTW:h150_robmiFMTW"; do
  lnk "$E/h150_cfl/stats/stats_${p#*:}.npz" "$F/h150/scores/stats_${p%%:*}.npz" || rc=1
done
O=$S/corrector_full; C=$F/full/corrector
mkdir -p "$C"/{logs,scores,tmp} "$F/full"/{test,logs}
for f in cov_oof_gv21.npy steps_gv21.npy dsc_gv21.json f11_train_gv21.npy; do lnk "$O/$f" "$C/$f" || rc=1; done
if [ "${NOMERGE:-0}" != 1 ]; then { [ -e "$F/feat_all/train/X.npy" ] && [ -e "$F/feat_all/test" ]; } || python "$CODE/f_merge.py" || rc=1; fi
ls "$F"/*/corrector | head -60
exit $rc
