#!/bin/bash
# ch1_package_final — FULL retrain of the FcCX base members (same commands as e_cx/cx_full_train.sh, output root = $D):
#   FM 11 psi nets (fm_full.json: 5 d30aux + 3 tower + 3 K120, 40 epochs, metric loss), 4 shards on 4 GPUs
#   TW tR_w9d seeds 0-3 (40 epochs OneCycle, last weights, spike guard), one GPU each
#   SC bl_d10: 17 members / 7 families
# on the mirror-canonical FEATURE-V2.2 cache $FEAT (train + holdout splits = all 7,041 labelled shots; scaler from the 6,891-shot train split).
set -uo pipefail
: "${D:?}" "${FEAT:?}" "${S11:?}" "${C2:?}" "${PK:?}"
export FEAT
mkdir -p "$D/frame_mlp/logs" "$D/temporal" "$D/scalars/logs"
J=${SLURM_JOB_ID:-x}; T0=$(date +%s); NG=${NGPU:-4}
cd "$C2/vendor/ch1_frame_mlp"
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i % NG)) METRIC=$S11/metric OUT=$D/frame_mlp python fm_train.py --configs "$PK/fm_full.json" --shard $i/4 > "$D/frame_mlp/logs/train_${J}_g$i.log" 2>&1 &
done
wait
echo "--- [$(( $(date +%s) - T0 ))s] FM done"; grep -h -E "FAILED|Traceback|Error" "$D"/frame_mlp/logs/train_${J}_g*.log | cut -c1-200 | tail -5
cd "$C2/vendor/ch1_temporal"
pids=()
for s in 0 1 2 3; do
  ( CUDA_VISIBLE_DEVICES=$((s % NG)) MET=$S11/metric OUT=$D/temporal FULL=1 TTAG=tR T_ARCH=res WTAG=w9d W_DROP=0.1 EP_W=40 STAGES=win \
      SEEDS=$s RUN_NAME=pf_full_tR_w9d_s$s python win_train.py > "$D/temporal/train_${J}_s$s.log" 2>&1; echo "TW full seed $s rc=$?" ) &
  pids+=($!)
done
cd "$C2/vendor/ch1_scalars"
M=""; for s in 0 1 2 3 4; do M+="la1_d10:$s,"; done
for s in 0 1; do for c in la1_d10_e45 la1_d10wd la1_d20 la1_d10_phys ra1_d10; do M+="$c:$s,"; done; done
sc() { local tasks=$1 nw=$2 i ps=()
  for i in $(seq 0 $((nw - 1))); do
    CUDA_VISIBLE_DEVICES=$((i % NG)) OUT=$D/scalars FULL=1 TASKS=$tasks WORKER=$i NWORK=$nw OMP_NUM_THREADS=4 python sc_nn.py >> "$D/scalars/logs/nn_${J}_w$i.log" 2>&1 & ps+=($!)
  done
  for i in "${ps[@]}"; do wait "$i" || echo SC_WORKER_FAILED; done; }
( sc "$M" 4; sc "gru_d10_fz:0,gru_d10_fz:1" 2; echo "SC full done [$(( $(date +%s) - T0 ))s]" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo "--- [$(( $(date +%s) - T0 ))s] TW + SC done"
echo "members: FM $(ls "$D"/frame_mlp/members | wc -l) / 11, TW $(ls "$D"/temporal/full/weights/*.pt | wc -l) / 4, SC $(ls "$D"/scalars/weights/full/*.pt | wc -l) / 17"
[ "$(ls "$D"/frame_mlp/members | wc -l)" = 11 ] && [ "$(ls "$D"/temporal/full/weights/*.pt | wc -l)" = 4 ] && [ "$(ls "$D"/scalars/weights/full/*.pt | wc -l)" = 17 ] || { echo TRAIN_BASE_INCOMPLETE; exit 1; }
echo "TRAIN_BASE_DONE wall_s $(( $(date +%s) - T0 )) x 4 GPU"
