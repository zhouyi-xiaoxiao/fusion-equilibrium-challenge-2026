# Network stages shared by the synthetic-equilibrium blend profiles (syn_xa_p7li50, syn_xa_p7c80). Not a profile: it is sourced by a
# profile AFTER its base inverse profile, and needs SYN_W (weight of the network map in the blend).
#
# Recipe "XA" (fixed; chosen on DIII-D labels, see METHODS.md section 3):
#   1. synreal   network inputs of every MAST frame (coil / solenoid / P2 split currents, I_p, F_vac, 33 Thomson features: outer / inner
#                edge, five isotherm pairs, T_e peak / symmetric axis, electron beta) - the same feature function as for synthetic frames.
#   2. syngen    synthetic free-boundary equilibria of OUR OWN forward model (the INV_P7c model: cubic FF' family, p' shape s = 0.5,
#                cut-cell quadrature) driven by the input columns of the shots being predicted (transductive, label-free): per anchor
#                frame (every 2nd frame with I_p >= 50 kA and a measured outer Thomson edge) the measured coil currents and I_p, then 10
#                draws of the profile / closure latents from the inverse's own DIII-D-label-fitted priors, Thomson rendered on the frame's
#                real channels with three noise realisations; seeded per shot (seed 3), so the result does not depend on the sharding.
#   3. syntrain  residual MLP 512 x 4, 4-member ensemble, 60 epochs, on the X-point-bounded synthetic samples with all three noise
#                realisations (GPU, about 3 min on one GH200), outputs 48 PCA coefficients of (psi - psi_vac) / I_p + q95, beta_N + aux;
#                then the prediction of every MAST frame: psi = exact vacuum flux of the frame + I_p x network, frames without a measured
#                outer edge filled from their neighbours, Gaussian smoothing along the shot (sigma 1 frame) of the plasma part and of the
#                scalars.
#   4. post synblend  map blend psi = SYN_W x network + (1 - SYN_W) x the base inverse's map; q95 = weighted geometric mean, beta_N =
#                weighted mean of each component's DIII-D-calibrated scalars (network: ch2/constants/syn_xa/calib_Cxa32.json from its
#                DIII-D twin, ch2/train_syn_d3d.sh; inverse: ch2/constants/<base>/calib_<tag>.json); 3-frame median, validity window,
#                24-bit psi mantissa. The packed base file is the fallback for a shot without any valid network scalar (never needed so far).
# With SYN_MODEL_FILE=<model .pt> the training is skipped and that model is used (e.g. the published weights of a Final run).
# Device and threads of syntrain: the trainer and the predictor use the GPU when torch sees one (the Development and rehearsal runs:
# one GH200 GPU, OMP_NUM_THREADS=8) and the CPU otherwise. SYN_THREADS=<n> sets the CPU threads (OMP / MKL / OpenBLAS); default 8 with
# a GPU (unchanged setting of the shipped files), all cores of the job without one (CPU fallback: about 15 min of training on 72 cores).
SYN_DIR=$WS/stage11_run/j_synth_net
EXTRA_CONST=$FEC_REPO/ch2/constants/syn_xa
SYN_NET_CALIB=$EXTRA_CONST/calib_Cxa32.json
SYN_CFG_REAL=(--cfg "$WS/stage11_run/ch2_G1/configs/cfg_soft_k50.json" --cfg "$WS/stage11_run/g_profile/configs/p7_base.json")
SYN_CFG_GEN=("${SYN_CFG_REAL[@]}" --cfg "$WS/stage11_run/g_profile/configs/p7c_final.json")
SYN_GEN_SET=(--set frame_stride=2 --set nsamp=10 --set ts_fit=true)
SYN_SEED=3
SYN_NAME=mCxa32
SYN_TRAIN_ARGS=(--xpt-only --reals 0,1,2)
SYN_PRED_ARGS=(--fill-nots --smooth 1 --smooth-scal)
PSTAGES=(synreal syngen syntrain)
POST=(synblend "${POST[@]}")
RUN_MODELS=("syn/models/model_$SYN_NAME.pt")

syn_nshots() { grep -c . "$1/shards/all.txt"; }
syn_threads_env() {  # -> the environment assignments for syn_train.py / syn_predict.py (see the header)
  local t
  if [ -n "${SYN_THREADS:-}" ]; then t=$SYN_THREADS
  elif [ "$($PYS -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null)" = 1 ]; then echo "OMP_NUM_THREADS=8"; return
  else t=${SLURM_CPUS_PER_TASK:-$(nproc)}; fi
  echo "OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t"
}
syn_summary_ok() {  # <summary json> -> exit 0 if it exists and has no fatal shot
  [ -s "$1" ] && $PYS -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('n_fatal', 1) == 0 else 1)" "$1"
}
stage_synreal() {  # <run dir> <mast shot dir>
  local O=$1 MD=$2 D=$1/syn
  if syn_summary_ok "$D/real/summary_mast_all.json" && [ "$(ls "$D/real/mast_all" 2>/dev/null | wc -l)" = "$(syn_nshots "$O")" ]; then echo "synreal: done"; return 0; fi
  ( cd "$SYN_DIR" && $PY syn_gen.py --mode real --machine mast --data-dir "$MD" --wall "$GEOM/mast_limiter_mask.npz" "${SYN_CFG_REAL[@]}" \
      --tag mast_all --out "$D" --procs "$NPROC" ) > "$O/logs/synreal.log" 2>&1
  grep -E "SUMMARY|FATAL|Traceback" "$O/logs/synreal.log" | cut -c1-300
  syn_summary_ok "$D/real/summary_mast_all.json"
}
stage_syngen() {  # <run dir> <mast shot dir>; NSHARD / SHARD as for solve
  local O=$1 MD=$2 D=$1/syn N=${NSHARD:-1} k s ks
  if [ -n "${SHARD:-}" ]; then ks=$SHARD; else ks=$(seq 0 $((N - 1))); fi
  for k in $ks; do
    s=$(printf %02d "$k")
    if syn_summary_ok "$D/synth/summary_mast_C2_s$s.json"; then echo "syngen shard $k: done"; continue; fi
    ( cd "$SYN_DIR" && $PY syn_gen.py --mode synth --machine mast --data-dir "$MD" --wall "$GEOM/mast_limiter_mask.npz" "${SYN_CFG_GEN[@]}" \
        "${SYN_GEN_SET[@]}" --tag mast_C2 --out "$D" --seed "$SYN_SEED" --shard "$k:$N" --procs "$NPROC" ) > "$O/logs/syngen_$k.log" 2>&1
    grep -E "SUMMARY|FATAL|Traceback" "$O/logs/syngen_$k.log" | cut -c1-300
    syn_summary_ok "$D/synth/summary_mast_C2_s$s.json" || { echo "SYNGEN_INCOMPLETE shard $k: run the same command again"; return 1; }
  done
}
stage_syntrain() {  # <run dir> <mast shot dir>; GPU if visible (training + prediction), else CPU with all cores of the job
  local O=$1 D=$1/syn M=$1/syn/models/model_$SYN_NAME.pt n
  n=$(ls "$D/synth/mast_C2" 2>/dev/null | grep -c '\.npz$')
  [ "$n" = "$(syn_nshots "$O")" ] || { echo "syntrain: synthetic set incomplete ($n of $(syn_nshots "$O") shots; run stage syngen)"; return 1; }
  syn_summary_ok "$D/real/summary_mast_all.json" || { echo "syntrain: run stage synreal first"; return 1; }
  mkdir -p "$D/models"
  local TENV; TENV=$(syn_threads_env)
  echo "syntrain: device $($PYS -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null), threads: $TENV"
  if [ ! -s "$M" ]; then
    if [ -n "${SYN_MODEL_FILE:-}" ]; then cp "$SYN_MODEL_FILE" "$M"; echo "syntrain: model taken from $SYN_MODEL_FILE (not retrained)"
    else
      ( cd "$SYN_DIR" && env $TENV $PYS syn_train.py --synth "$D/synth/mast_C2" --out "$D/models" --name "$SYN_NAME" "${SYN_TRAIN_ARGS[@]}" ) > "$O/logs/syntrain.log" 2>&1
      grep -E "loaded|VAL real0|saved|Error|Traceback" "$O/logs/syntrain.log" | cut -c1-300
    fi
  fi
  [ -s "$M" ] || { echo "syntrain: no model"; return 1; }
  echo "model $(sha256sum "$M" | cut -c1-64) $M"
  if [ ! -s "$D/pred/files_${SYN_NAME}fg1_mast.txt" ]; then
    ( cd "$SYN_DIR" && env $TENV $PYS syn_predict.py --model "$M" --real "$D/real/mast_all" --machine mast --tag "${SYN_NAME}fg1" --out "$D/pred" \
        "${SYN_PRED_ARGS[@]}" ) > "$O/logs/synpredict.log" 2>&1
    grep -E "PREDICT|Error|Traceback" "$O/logs/synpredict.log" | cut -c1-300
  fi
  [ "$(grep -c . "$D/pred/files_${SYN_NAME}fg1_mast.txt" 2>/dev/null)" = "$(syn_nshots "$O")" ]
}
post_synblend() {  # <run dir> <input npz = packed base inverse file> <output npz> <mast shot dir>
  local O=$1 IN=$2 OUTF=$3 D=$1/syn WB
  WB=$($PYS -c "print(1-$SYN_W)")
  rm -rf "$D/blend" && mkdir -p "$D/blend"
  ( cd "$SYN_DIR" && $PYS syn_avg.py --runs "$D/pred:${SYN_NAME}fg1:$SYN_NET_CALIB" "$O:$TAG:$CONST/calib_$TAG.json" --weights "$SYN_W,$WB" \
      --out "$D/blend" --tag mbl --machine mast ) 2>&1 | grep -E "AVG|Error|Traceback|assert" | cut -c1-300
  ( cd "$SYN_DIR" && $PYS syn_pack.py --parts "$D/blend:mbl" --q95 "net:$D/blend:mbl:$D/blend/calib_mbl.json" --betaN "net:$D/blend:mbl:$D/blend/calib_mbl.json" \
      --base-sub "$IN" --name blend --out-dir "$D/blend" ) 2>&1 | grep -E "PACK|Error|Traceback|assert" | cut -c1-400
  [ -s "$D/blend/submission/blend/mast_public_test.npz" ] && cp "$D/blend/submission/blend/mast_public_test.npz" "$OUTF"
}
