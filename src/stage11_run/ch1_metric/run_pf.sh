#!/bin/bash
# per-frame Jacobians: merge shards -> holdout predictions + oracle interventions -> official harness -> report
set -uo pipefail
STEPS=${STEPS:-merge+prep+score+report}
if [[ "+$STEPS+" == *+merge+* ]]; then python perframe.py merge || { echo MERGE_FAILED; exit 1; }; fi
if [[ "+$STEPS+" == *+prep+* ]]; then python perframe.py prep || { echo PFPREP_FAILED; exit 1; }; fi
if [[ "+$STEPS+" == *+score+* ]]; then
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  while read -r name; do
    [ -n "$name" ] || continue
    lg=$OUT/hold/scores/score_$name.log
    grep -q COMPOSITE "$lg" 2>/dev/null && { echo "already scored: $name"; continue; }
    python "$WS/stage2_run/score_stage2.py" --machine d3d --gt-dir "$WS/scaleup_run/data/diii_d_train" \
      --skip 1000 --n 150 --pred "$OUT/hold/preds_d3d_$name.npz" --tag "$name" > "$lg" 2>&1 &
  done < "$OUT/hold/new_names_pf.txt"
  wait
  grep -h COMPOSITE "$OUT"/hold/scores/score_PF_*.log
fi
if [[ "+$STEPS+" == *+report+* ]]; then python perframe.py report || { echo PFREPORT_FAILED; exit 1; }; fi
if [[ "+$STEPS+" == *+selftest+* ]]; then python metric_loss.py selftest || { echo SELFTEST_FAILED; exit 1; }; fi
