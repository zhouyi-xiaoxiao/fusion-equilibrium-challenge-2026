#!/bin/bash
# DIII-D twin of the synthetic-equilibrium network (recipe "XA", ch2/pipelines/syn_net.inc): regenerates the two DIII-D calibration
# constants of the network, ch2/constants/syn_xa/calib_Cxa32.json (C_q95, C_betaN). DIII-D TRAIN labels only; no MAST file is read.
#
#   [NSHARD=N SHARD=k] bash ch2/train_syn_d3d.sh <diii_d_train_dir> <run_name> [stage ...]       stages: gen real train calib (default all)
#
#   gen    synthetic equilibria of our own forward model (INV_P7c model) driven by DIII-D TRAIN shots: holdout (150) and the 150 shots of
#          splits/d3d_syn_genban.txt excluded, every 4th remaining shot (1,686 drivers), 20 random anchor frames x 10 latent draws, the
#          MAST-like pseudo-horizontal Thomson chord, three noise realisations; seeded per shot (seed 3) -> independent of the sharding.
#          CPU only; about 230 core-h (12 x 144 cores, 8 min each). NSHARD / SHARD split it into independent jobs.
#   real   network inputs of the 300 DIII-D calibration shots x 8 frames (the same shots / frames as the pcal stage of the inverses)
#   train  the twin network Cxa32: same architecture / recipe as the MAST network, trained on this DIII-D synthetic set without the 120
#          shots of splits/d3d_syn_trainban.txt (1,654 drivers remain; GPU, about 3 min on one GH200)
#   calib  C_q95 = median(label q95 / network q95), C_betaN = least squares of label beta_N on network beta_N, on the calibration frames;
#          prints COMPARE lines against ch2/constants/syn_xa/calib_Cxa32.json
# The twin network itself is not used for MAST; only its two constants are. Output: $SCR/ch2_syn_d3d/<run_name>/
set -uo pipefail
. "$(dirname "$0")/../env.sh"; . "$FEC_REPO/ch2/common.sh"
DD=$(readlink -f "${1:?usage: train_syn_d3d.sh <diii_d_train_dir> <run_name> [stages]}"); NAME=${2:?run name}; shift 2; STAGES=${*:-gen real train calib}
O=$SCR/ch2_syn_d3d/$NAME; mkdir -p "$O/logs"; T0=$(date +%s)
names_dir "$FEC_REPO/splits/d3d_subset1150.txt" "$SCR/splits/d3d_subset1150"; HOLD=$SCR/splits/d3d_subset1150:1000:1150
SYN_DIR=$WS/stage11_run/j_synth_net
CFG_A=(--cfg "$WS/stage11_run/ch2_G1/configs/cfg_soft_k50.json" --cfg "$WS/stage11_run/g_profile/configs/p7_base.json")
CFG_C=("${CFG_A[@]}" --cfg "$WS/stage11_run/g_profile/configs/p7c_final.json")
REF=$FEC_REPO/ch2/constants/syn_xa/calib_Cxa32.json
N=${NSHARD:-1}
ok() { [ -s "$1" ] && $PYS -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('n_fatal', 1) == 0 else 1)" "$1"; }
echo "=== ch2 train_syn_d3d $(hostname) $(date -u) train_dir=$DD ($(ls "$DD" | grep -c parquet) files) stages=$STAGES nshard=$N shard=${SHARD:-all}"
for st in $STAGES; do case $st in
gen)
  if [ -n "${SHARD:-}" ]; then ks=$SHARD; else ks=$(seq 0 $((N - 1))); fi
  for k in $ks; do
    s=$(printf %02d "$k"); ok "$O/synth/summary_d3d_C2_s$s.json" && { echo "gen shard $k: done"; continue; }
    ( cd "$SYN_DIR" && $PY syn_gen.py --mode synth --machine d3d --data-dir "$DD" --exclude-holdout "$HOLD" --disjoint-from "$FEC_REPO/splits/d3d_syn_genban.txt" \
        --stride 4 --wall "$GEOM/d3d_limiter_mask.npz" "${CFG_C[@]}" --set n_anchor=20 --set frame_stride=1 --set nsamp=10 --set ts_fit=true \
        --tag d3d_C2 --out "$O" --seed 3 --shard "$k:$N" --procs "$NPROC" ) > "$O/logs/gen_$k.log" 2>&1
    grep -E "SUMMARY|FATAL|Traceback" "$O/logs/gen_$k.log" | cut -c1-300
    ok "$O/synth/summary_d3d_C2_s$s.json" || { echo "GEN_INCOMPLETE shard $k"; exit 1; }
  done ;;
real)
  ( cd "$SYN_DIR" && $PY syn_gen.py --mode real --machine d3d --data-dir "$DD" --exclude-holdout "$HOLD" --stride 23 --wall "$GEOM/d3d_limiter_mask.npz" \
      "${CFG_A[@]}" --set frames_per_shot=8 --set "stress={}" --tag d3d_pcal --out "$O" --procs "$NPROC" ) > "$O/logs/real.log" 2>&1
  grep -E "SUMMARY|FATAL|Traceback" "$O/logs/real.log" | cut -c1-300
  ok "$O/real/summary_d3d_pcal.json" || { echo "REAL_FAILED (see $O/logs/real.log)"; exit 1; } ;;
train)
  n=$(ls "$O/synth/d3d_C2" 2>/dev/null | grep -c '\.npz$'); echo "synthetic set: $n driver shots"
  [ -s "$O/models/model_Cxa32.pt" ] || ( cd "$SYN_DIR" && OMP_NUM_THREADS=8 $PYS syn_train.py --synth "$O/synth/d3d_C2" --out "$O/models" --name Cxa32 \
      --ban "$FEC_REPO/splits/d3d_syn_trainban.txt" --xpt-only --reals 0,1,2 ) > "$O/logs/train.log" 2>&1
  grep -E "loaded|VAL real0|saved|Error|Traceback" "$O/logs/train.log" | cut -c1-300
  [ -s "$O/models/model_Cxa32.pt" ] || { echo "TRAIN_FAILED: no model (see $O/logs/train.log); run the same command again"; exit 1; }
  echo "model $(sha256sum "$O/models/model_Cxa32.pt" | cut -c1-64)" ;;
calib)
  ( cd "$SYN_DIR" && OMP_NUM_THREADS=8 $PYS syn_predict.py --model "$O/models/model_Cxa32.pt" --real "$O/real/d3d_pcal" --machine d3d --tag Cxa32_pcal \
      --out "$O/pred" --calib-fit "$O/pred/calib_Cxa32.json" ) 2>&1 | grep -E "CALIB|Error|Traceback" | cut -c1-600
  [ -s "$O/pred/calib_Cxa32.json" ] || { echo "CALIB_FAILED: no calibration file"; exit 1; }
  $PYS - "$O/pred/calib_Cxa32.json" "$REF" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for k in ("C_q95", "C_betaN", "n_q", "n_b"):
    print(f"COMPARE {k:8s} new {a[k]!r} shipped {b[k]!r} {'IDENTICAL' if a[k] == b[k] else 'rel diff %.3e' % (a[k] / b[k] - 1)}")
PY
  ;;
*) echo "unknown stage $st"; exit 2;; esac; echo "--- [$(( $(date +%s) - T0 ))s] $st"; done
echo "CH2_SYN_D3D_DONE $O"
