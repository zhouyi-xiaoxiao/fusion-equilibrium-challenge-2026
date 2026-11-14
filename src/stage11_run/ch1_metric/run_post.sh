#!/bin/bash
# build M (train only) -> holdout sanity: predicted table, official harness on new variants, report.
set -uo pipefail
STEPS=${STEPS:-build+prep+score+report}
if [[ "+$STEPS+" == *+build+* ]]; then python build_metric.py || { echo BUILD_FAILED; exit 1; }; fi
if [[ "+$STEPS+" == *+prep+* ]]; then python holdout_check.py prep || { echo PREP_FAILED; exit 1; }; fi
if [[ "+$STEPS+" == *+score+* ]]; then
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  t0=$(date +%s)
  while read -r name; do
    [ -n "$name" ] || continue
    lg=$OUT/hold/scores/score_$name.log
    grep -q COMPOSITE "$lg" 2>/dev/null && { echo "already scored: $name"; continue; }
    python "$WS/stage2_run/score_stage2.py" --machine d3d --gt-dir "$WS/scaleup_run/data/diii_d_train" \
      --skip 1000 --n 150 --pred "$OUT/hold/preds_d3d_$name.npz" --tag "$name" > "$lg" 2>&1 &
  done < "$OUT/hold/new_names.txt"
  wait
  echo "SCORE_WALL_MIN $(( ($(date +%s)-t0)/60 ))"
  grep -h COMPOSITE "$OUT"/hold/scores/score_*.log
fi
if [[ "+$STEPS+" == *+report+* ]]; then python holdout_check.py report || { echo REPORT_FAILED; exit 1; }; fi
if [ -n "${PF_STEPS:-}" ]; then STEPS=$PF_STEPS bash ./run_pf.sh; fi
