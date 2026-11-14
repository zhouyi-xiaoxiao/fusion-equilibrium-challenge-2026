#!/bin/bash
# Model-agnostic consistency corrector — apply to ANY prediction file in the harness layout.
# Run inside an sbatch allocation (72 cores; no GPU needed). Env:
#   PRED    prediction npz (shot_XXXX_{psirz,q95,betaN}, shots in the order of $CC_FEAT/$SPLIT/offsets.npy)   or
#   COEF    PCA80 coefficient npy (N,80) + QB = harness npz to borrow q95/betaN from
#   TAG     short name of the base (used in every output name)
#   SPLIT   holdout | test | any sub-directory of $CC_FEAT holding X.npy offsets.npy times.npy   (default holdout)
#   FAM     model family whose OOF statistics are used (cov/steps/fusion weights): gv21 (default) | inc
#   TGT     target fusion: rob (default) | dlam | direct
#   CFG     config for non-holdout splits (default r1e+09_e1_i13 = no trust clip, eta 1, 3 frozen-J steps + relinearise)
#   DIRECT  direct-head tag (default direct2), DSEEDS its seeds (default 0,1,2,3,4,5)
#   BLEND_FROZEN  blend json whose constants are reused (final refit: no honest validation slice)   [default: fit]
#   COV / DSC     family covariance npy / innovation-scale json (default cov_oof_$FAM.npy / heads_innov_$FAM.json)
#   JREUSE=1      reuse existing f11/jac/der files of this TAG instead of recomputing them
#   KEEP_BASE     comma list of scalars whose target stays the base's own value (gate; e.g. volume,Rmin for frame_mlp bases)
# Holdout: scores the base + a small (eta, iters) grid with the emulated official composite, writes the selected
#          corrected files preds_d3d_cc_<TAG>_<TGT>_<cfg>{,_A,_B}.npz and (SCORE=1) runs the official harness on them.
# Other splits: writes preds_<SPLIT>_<TAG>_<TGT>_<CFG>.npz.
set -uo pipefail
: "${TAG:?}"; SPLIT=${SPLIT:-holdout}; FAM=${FAM:-gv21}; TGT=${TGT:-rob}; CFG=${CFG:-r1e+09_e1_i13}
DIRECT=${DIRECT:-direct2}; DSEEDS=${DSEEDS:-0,1,2,3,4,5}; NW=${NW:-64}; SCORE=${SCORE:-1}
WS=${WS:?set WS}
L=$CC_OUT/logs; mkdir -p "$L" "$CC_OUT/scores"
cd "$WS/stage11_run/ch1_corrector"
[ "$FAM" = inc ] && FOOF=incoof || FOOF=$FAM
COV=${COV:-cov_oof_$FAM.npy}; DSC=${DSC:-heads_innov_$FAM.json}; BF=${BLEND_FROZEN:+--frozen $BLEND_FROZEN}; KB=${KEEP_BASE:+--keep-base $KEEP_BASE}
if [ -n "${COEF:-}" ]; then SRC="--coef $COEF"; QB=${QB:+--qb $QB}; else SRC="--pred $PRED"; QB=""; fi
if [ "$SPLIT" = holdout ] && ! { [ "${JREUSE:-0}" = 1 ] && [ -e "$CC_OUT/der_hold_$TAG.npz" ]; }; then
  python cc_derive.py $SRC --tag "$TAG" --split holdout --workers $NW > "$L/derive_hold_$TAG.log" 2>&1 || { echo DERIVE_FAILED; exit 1; }
  grep "^emul" "$L/derive_hold_$TAG.log"
fi
if [ "${JREUSE:-0}" = 1 ] && [ -e "$CC_OUT/jac_${SPLIT}_$TAG.npz" ] && [ -e "$CC_OUT/f11_${SPLIT}_$TAG.npy" ]; then echo "JREUSE jac_${SPLIT}_$TAG"; else
python cc_jac.py --split "$SPLIT" --tag "$TAG" $SRC --steps "$CC_OUT/steps_$FAM.npy" --workers $NW > "$L/jac_${SPLIT}_$TAG.log" 2>&1 || { echo JAC_FAILED; exit 1; }
tail -1 "$L/jac_${SPLIT}_$TAG.log"; fi
DP=$CC_OUT/hp_${DIRECT}_mlp_$SPLIT.npy
[ -e "$DP" ] || OMP_NUM_THREADS=32 python cc_heads.py predict --tag "$DIRECT" --seeds "$DSEEDS" --model mlp --feat-dir "$CC_FEAT/$SPLIT" --out "$DP"
if [ "$TGT" = direct ]; then DH=$DP; else
  python cc_blend.py --base "$TAG" --split "$SPLIT" --base-train "f11_train_$FOOF.npy" --direct "$DIRECT" $BF $KB | tee "$L/blend_${SPLIT}_$TAG.log"
  [ "$SPLIT" = holdout ] && DH=tgt_${TAG}_$TGT.npy || DH=tgt_${SPLIT}_${TAG}_$TGT.npy
fi
COMMON="--split $SPLIT --tag $TAG $SRC $QB --name $TGT --dhat $DH --direct --targets 11 --cov $COV --dsc $DSC --workers $NW"
if [ "$SPLIT" = holdout ]; then
  python cc_apply.py $COMMON --rs 1e9 --etas ${ETAS:-0.5,1} --iters ${ITERS:-3,13} --write sel > "$L/apply_${TAG}_$TGT.log" 2>&1 || { echo APPLY_FAILED; tail -5 "$L/apply_${TAG}_$TGT.log"; exit 1; }
  grep -v "^WROTE" "$L/apply_${TAG}_$TGT.log" | cut -c1-400
  if [ "$SCORE" = 1 ]; then
    sc() { ( OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python "$WS/stage2_run/score_stage2.py" --machine d3d --gt-dir "$WS/scaleup_run/data/diii_d_train" --skip "$3" --n "$4" --pred "$1" --tag "$2" > "$CC_OUT/scores/score_$2.log" 2>&1 ) & }
    for f in "$CC_OUT"/preds_d3d_cc_${TAG}_${TGT}_*.npz; do
      case "$f" in *_A.npz|*_B.npz) continue;; esac
      t=$(basename "$f" .npz); t=${t#preds_d3d_}
      sc "$f" "$t" 1000 150; sc "${f%.npz}_A.npz" "${t}_A" 1000 75; sc "${f%.npz}_B.npz" "${t}_B" 1075 75
    done
    if [ -n "${PRED:-}" ] && [ "${SCORE_BASE:-1}" = 1 ]; then
      cp -f "$PRED" "$CC_OUT/preds_d3d_base_$TAG.npz"; python cc_split.py "$CC_OUT/preds_d3d_base_$TAG.npz"
      sc "$CC_OUT/preds_d3d_base_$TAG.npz" "base_$TAG" 1000 150; sc "$CC_OUT/preds_d3d_base_${TAG}_A.npz" "base_${TAG}_A" 1000 75; sc "$CC_OUT/preds_d3d_base_${TAG}_B.npz" "base_${TAG}_B" 1075 75
    fi
    [ "${NOWAIT:-0}" = 1 ] || wait
  fi
else
  python cc_apply.py $COMMON --rs 1e9 --etas 1 --iters "${CFG##*_i}" --write "$CFG" > "$L/apply_${SPLIT}_${TAG}_$TGT.log" 2>&1 || { echo APPLY_FAILED; tail -5 "$L/apply_${SPLIT}_${TAG}_$TGT.log"; exit 1; }
  cat "$L/apply_${SPLIT}_${TAG}_$TGT.log" | cut -c1-400
fi
