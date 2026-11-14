#!/bin/bash
# Stage 11F / corr_canon — retrain the consistency-corrector DIRECT HEADS on the mirror-canonical (cfl) feature caches.
# Unchanged recipe (cc_heads.py, ch1_corrector): residual MLP 768x4, AdamW 5e-4, cosine, EMA 0.995, bf16, 6 seeds;
#   direct2   = dev style (5 % shot-disjoint val slice, early stopping, max 45 epochs) -> fits the rob constants on that slice
#               with the (unchanged) gv21 OOF maps as the base proxy, exactly as f2_run_cc_b1.sh / job_cc2.sbatch;
#   direct2fx = shipped recipe (all train shots, fixed 42 epochs) -> used for the correction.
#   FULL      = cc_heads --full 42 epochs on all 7,041 shots (tag direct2, as corrector_full/job_full.sbatch).
# Only the head INPUTS change: CC_FEAT = canonical cache (430 odd columns x sign(Ip_shot), raw sign flag kept, scaler refit).
#   sbatch -J fc_heads job_gpu.sbatch bash f_heads.sh
set -uo pipefail
bash "$CODE/f_prep.sh" || { echo PREP_FAILED; exit 1; }
CCD=$WS/stage11_run/ch1_corrector
one() {  # set gpu
  local s=$1 g=$2 L=$F/$1/corrector/logs
  export CC_OUT=$F/$s/corrector
  case $s in fold3|fold2) export CC_FEAT=$E/${s}_cfl/feat;; h150) export CC_FEAT=$E/h150_cfl/feat;; full) export CC_FEAT=$F/feat_all;; esac
  cd "$CCD"
  if [ "$s" = full ]; then
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 python cc_heads.py train --tag direct2 --seeds 0,1,2,3,4,5 --epochs 42 --no-gru --full > "$L/heads_direct2.log" 2>&1
    echo "rc=$?" >> "$L/heads_direct2.log"
  else
    ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 python cc_heads.py train --tag direct2 --seeds 0,1,2,3,4,5 --epochs 45 --no-gru > "$L/heads_direct2.log" 2>&1; echo "rc=$?" >> "$L/heads_direct2.log" ) &
    ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 python cc_heads.py train --tag direct2fx --seeds 0,1,2,3,4,5 --epochs 42 --no-gru --full > "$L/heads_direct2fx.log" 2>&1; echo "rc=$?" >> "$L/heads_direct2fx.log" ) &
    wait
  fi
}
( one fold3 0 ) & ( one fold2 1 ) & ( one h150 2 ) & ( one full 3 ) &
wait
for s in fold3 fold2 h150 full; do echo "== heads $s"; grep -h "seed [0-9]*:\|rc=\|HOLDOUT\|Traceback\|Error" "$F/$s"/corrector/logs/heads_direct2*.log | cut -c1-260; done
grep -L "rc=0" "$F"/*/corrector/logs/heads_direct2*.log | grep . && { echo HEADS_FAILED; exit 1; }
# ---- rob constants, same procedure as the incumbent (val slice of train, gv21 OOF maps as the base proxy, canonical direct2 heads)
cd "$CCD"
for s in fold3 fold2; do
  export CC_OUT=$F/$s/corrector CC_FEAT=$E/${s}_cfl/feat
  python cc_blend.py --base gv21 --split train --base-train f11_train_gv21.npy --direct direct2 | tee "$CC_OUT/logs/blend_constants.log"
  cp -f "$CC_OUT/blend_train_gv21.json" "$CC_OUT/blend_frozen_f2c.json"; rm -f "$CC_OUT"/tgt_train_gv21_*.npy
  echo "   OLD (raw heads): $(head -2 "$S/$s/corrector/logs/blend_constants.log" | tr '\n' ' ' | cut -c1-400)"
done
export CC_OUT=$F/h150/corrector CC_FEAT=$E/h150_cfl/feat
python cc_blend.py --base gv21 --split holdout --base-train f11_train_gv21.npy --direct direct2 | tee "$CC_OUT/logs/blend_constants.log"
cp -f "$CC_OUT/blend_holdout_gv21.json" "$CC_OUT/blend_frozen_h150c.json"
cp -f "$CC_OUT/blend_holdout_gv21.json" "$F/full/corrector/blend_frozen_gv21c.json"
python "$CODE/f_headdiag.py" | tee "$F/logs/headdiag.txt"
md5sum "$F"/*/corrector/blend_frozen_*c.json "$F"/*/corrector/ck_direct2*_s0.pt
