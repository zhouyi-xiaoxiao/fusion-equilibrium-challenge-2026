#!/bin/bash
# DIII-D profile xhc — training stage FT: the label-polygon contour fine-tune of the base's 11 FM members and 4 TW seeds
# (METHODS.md section "Profile xhc"). Settings fixed in development: mode lab, nominal share 5, cap 20, lr 2e-4 -> 2e-5 cosine,
# 40 epochs, EMA 0.999 from the warm start, median calibration of the multiplier (FM), the members' own seeds (fine-tune replicate 0).
#
#   [USE_SLURM=1] [AFTER=<job id>] [CMP=1] bash ch1/train_xhc_ft.sh <BASE_ROOT> <OUT_ROOT> <diii_d_train dir>
#
# BASE_ROOT  the FULL base the fine-tune starts from: frame_mlp/members/FULL_*/model.pt, temporal/full/weights/tR_w9d_s{0..3}.pt and
#            scalars/weights/full/*.pt = MODEL_ROOT/base of ch1/train.sh, or the shipped base $SCR/stage11_ch1/e_cx/full
# OUT_ROOT   OUT_ROOT/liloss  reference contours of the training labels (CPU; ch1d_liloss/li_prep.py)
#            OUT_ROOT/ft      c0 (warm-start coefficients on the training rows, GPU forward), anchor bank + shot weights + 256 held-out
#                             gate rows (CPU; jc_bank.py), label bank (CPU; jc_bank_label.py), fine-tuned members (GPU; jc_ft.py,
#                             jc_ft_tw.py)
#            OUT_ROOT/xhc     the base root the predict path of profile xhc loads: the 11 + 4 fine-tuned files and copies of the 17 SC
#                             files of BASE_ROOT (predict: CH1_PROFILE=xhc BASE_ROOT=OUT_ROOT/xhc ... bash ch1/predict.sh)
# diii_d_train dir   the public diii_d_train parquet files: labels of the 7,041 labelled shots (training labels only; read by the two
#            CPU label caches). Every other input is generated data: weights/GENERATED.sha256 (canonical v2.2 cache, metric) and
#            weights/GENERATED_XHC.sha256 (the cache's two row-order file lists), PCA bases of weights/MANIFEST.sha256.
# USE_SLURM=1: li cache (1 CPU node) at once; c0 (1 GPU) after AFTER; anchor bank and label bank (1 CPU node each); FM fine-tune
#   (4 jobs x 1 GPU, 11 members); TW fine-tune (4 jobs x 1 GPU, one seed each); assemble (+ CMP) after all of them.
#   Local mode: the same steps one after the other on this machine, NGPU GPUs (default 1).
# CMP=1 compares OUT_ROOT/xhc with weights/MANIFEST_CH1_XHC.sha256 (sha256, then tensor equality) -> OUT_ROOT/xwcmp.json.
# Measured (development, GH200): FM 11 x 3.5 min, TW 4 x 17 min, c0 1 min of one GPU = 1.9 GPU-h; CPU about 1 node-h.
set -euo pipefail
. "$(dirname "$0")/../env.sh"
BASE=$(readlink -m "${1:?usage: train_xhc_ft.sh <BASE_ROOT> <OUT_ROOT> <diii_d_train dir>}"); OUT=$(readlink -m "${2:?OUT_ROOT}")
DTR=$(readlink -f "${3:?diii_d_train dir}")
S11=$SCR/stage11_ch1; CODE=$WS/stage11_run/j_dlcfs_contour; LIP=$WS/stage11_run/ch1d_liloss/li_prep.py
for d in "$S11/e_cx" "$S11/xhc"; do
  case "$OUT/" in "$(readlink -f "$d" 2>/dev/null || echo /nonexistent)"/*) echo "refusing to write into the shipped root $d"; exit 1;; esac
done
BIN=""; [ -n "${AFTER:-}" ] || BIN="$BASE/frame_mlp/members/FULL_d30aux_s0/model.pt $BASE/temporal/full/weights/tR_w9d_s3.pt"   # with AFTER the base is still being trained
for f in $BIN "$S11/e_cx/h150/feat/train/X.npy" "$S11/e_cx/h150/feat/holdout/X.npy" "$S11/e_cx/h150/feat/train/files.txt" \
         "$S11/e_cx/h150/feat/holdout/files.txt" "$S11/metric/L_80.npy" "$SCR/stage6_H/pca_80.npz"; do
  [ -s "$f" ] || { echo "missing input $f (generated data: MANIFESTS=\"weights/GENERATED.sha256 weights/GENERATED_XHC.sha256\" bash tools/fetch_weights.sh ...)"; exit 1; }; done
[ "$(ls "$DTR"/*.parquet | wc -l)" -ge 7041 ] || { echo "$DTR: expected the 7,041 diii_d_train parquet files"; exit 1; }
mkdir -p "$OUT"/{liloss,ft/logs,logs}
# the stage's environment (exported into every job)
XENV="export WS=$WS SCR=$SCR FEC_REPO=$FEC_REPO XFT_BASE=$BASE J=$OUT/ft LI=$OUT/liloss D3D_TRAIN_DIR=$DTR SET=full PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=$SCR/.mplcache XDG_CACHE_HOME=$SCR/.xdg_cache; . $WS/.venv/bin/activate; cd $CODE"
FM_ARGS="--modes lab --share 5.0 --cap 20.0 --lr 2e-4 --epochs 40 --calib median --labdiag --suffix _s500EM"
TW_ARGS="--modes lab --share 5.0 --cap 20.0 --lr 2e-4 --epochs 40 --suffix _s500E"
# each step: <command> > log; its status is the step's status (a failed step stops the local run / fails the job, so that dependent
# SLURM jobs never start on incomplete inputs)
step() { echo "$2 > $OUT/logs/$1.log 2>&1; rc=\$?; echo STEP_$1_RC \$rc; grep -h -E '$3' $OUT/logs/$1.log | tail -${4:-3} | cut -c1-500; [ \$rc = 0 ]"; }
LI_CMD=$(step li_prep "SPLITS=holdout,train WORKERS=\$NW OMP_NUM_THREADS=1 python $LIP" 'split|FAIL|Traceback|Error')
C0_CMD=$(step c0 "OMP_NUM_THREADS=8 python jc_c0.py" 'c0 done|Traceback|Error')
BK_CMD=$(step bank "OMP_NUM_THREADS=1 python jc_bank.py --set full --workers \$NW" 'ok_frac|Traceback|Error' 1)
LB_CMD=$(step banklab "SPLITS=holdout,train OMP_NUM_THREADS=1 python jc_bank_label.py --workers \$NW" 'ok_frac|Traceback|Error')
fm_cmd() { step ft_fm_g$1 "OMP_NUM_THREADS=8 python jc_ft.py $FM_ARGS --shard $1/4" 'DONE|FAILED|Traceback|finished' 4; }
tw_cmd() { step ft_tw_s$1 "OMP_NUM_THREADS=8 python jc_ft_tw.py $TW_ARGS --seeds $1" 'DONE|FAILED|Traceback|finished'; }
ASM_CMD=$(step assemble "python $FEC_REPO/ch1/xhc_assemble.py $BASE $OUT ${CMP:+--cmp $OUT/xwcmp.json}" 'XHC_|Error|missing')
T0=$(date +%s)
if [ "${USE_SLURM:-0}" = 1 ]; then
  P=${SLURM_PARTITION:-workq}
  sb() { local nm=$1 dep=$2 res=$3 cmd=$4; sbatch --parsable -J "$nm" -p "$P" -N 1 $res ${dep:+--dependency=afterok:$dep} -o "$OUT/logs/slurm_%x_%j.out" \
           --wrap "set -uo pipefail; $XENV; export NW=\${SLURM_CPUS_PER_TASK:-72} TMPDIR=/tmp/xft_\$SLURM_JOB_ID; mkdir -p \$TMPDIR; echo \"=== \$SLURM_JOB_NAME \$SLURM_JOB_ID \$(hostname) \$(date -u +%FT%TZ)\"; nvidia-smi -L 2>/dev/null | head -4; T=\$(date +%s); $cmd; rc=\$?; echo \"=== rc \$rc wall_s \$(( \$(date +%s) - T ))\"; exit \$rc"; }
  CPU="--cpus-per-task=288 --mem=440G -t 01:30:00"; GPU1="--gpus=1 --cpus-per-task=72 --mem=110G"
  JLI=$(sb xft_li "" "$CPU" "$LI_CMD")
  JC0=$(sb xft_c0 "${AFTER:-}" "$GPU1 -t 00:30:00" "$C0_CMD")
  JBK=$(sb xft_bank "$JLI:$JC0" "$CPU" "$BK_CMD")
  JLB=$(sb xft_banklab "$JLI" "$CPU" "$LB_CMD")
  JFT=""
  for i in 0 1 2 3; do JFT="$JFT:$(sb xft_fm$i "$JBK:$JLB" "$GPU1 -t 01:00:00" "$(fm_cmd $i)")"; done
  for s in 0 1 2 3; do JFT="$JFT:$(sb xft_tw$s "$JBK:$JLB" "$GPU1 -t 01:00:00" "$(tw_cmd $s)")"; done
  JAS=$(sb xft_asm "${JFT#:}" "--cpus-per-task=16 --mem=60G -t 00:30:00" "$ASM_CMD")
  J="$JLI $JC0 $JBK $JLB ${JFT//:/ } $JAS"; J=$(echo $J)
  echo "$J" > "$OUT/xft_jobs.txt"
  echo "XFT_SUBMITTED out=$OUT jobs: $J"
  echo "GPU-h when done: sacct -X -n -j ${J// /,} --format=JobID,JobName,Elapsed,AllocTRES%60"
  exit 0
fi
# local mode
NG=${NGPU:-1}; export NW=${NW:-$(nproc)}
run() { ( set +e +o pipefail; eval "$XENV"; export TMPDIR=${TMPDIR:-/tmp}; eval "$1" ); }
run "$LI_CMD"; run "CUDA_VISIBLE_DEVICES=0 $C0_CMD"; run "$BK_CMD"; run "$LB_CMD"
pids=(); k=0
wall() { local p; for p in "$@"; do wait "$p"; done; }     # every process must succeed (set -e)
for i in 0 1 2 3; do run "export CUDA_VISIBLE_DEVICES=$(( k % NG )); $(fm_cmd $i)" & pids+=($!); k=$((k + 1)); [ $(( k % NG )) -ne 0 ] || { wall "${pids[@]}"; pids=(); }; done
for s in 0 1 2 3; do run "export CUDA_VISIBLE_DEVICES=$(( k % NG )); $(tw_cmd $s)" & pids+=($!); k=$((k + 1)); [ $(( k % NG )) -ne 0 ] || { wall "${pids[@]}"; pids=(); }; done
[ ${#pids[@]} = 0 ] || wall "${pids[@]}"
run "$ASM_CMD"
echo "XFT_DONE out=$OUT wall_s $(( $(date +%s) - T0 )) on $NG GPU(s)"
