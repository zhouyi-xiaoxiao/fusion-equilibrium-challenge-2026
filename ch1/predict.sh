#!/bin/bash
# Ch1 inference entrypoint: DIII-D input shots (parquet, inputs only) -> submission .npz.
#
#   [CH1_PROFILE=<profile>] bash ch1/predict.sh <d3d_shot_dir> <run_name> [<stream_order.txt>|sorted] [<output file name>]
#
# CH1_PROFILE selects the profile ch1/profiles/<name>.sh (default: ch1/profiles/DEFAULT): fccx | hc10x24 (ch1/profiles/README.md).
# Output: $SCR/stage11_ch1/final_path/runs/<run_name>/submission.npz (+ a copy under <output file name> if given), gate.json, gate.log,
# profile.txt. The full list of stages, artefacts and options is in the header of src/stage11_run/ch1_package_final/predict.sh.
# The log of a correct run ends with "HEADS_CHECK OK: profile <name> ..." (and, for hc10x24, contains the stage line
# "heads v2.2 c10 (24 seeds)"): always run a profile through THIS entrypoint, which hands the profile's HEADS to the inner script.
# USE_SLURM=1 submits the run as one SLURM job (SLURM_PARTITION, PRED_GPUS, PRED_CPUS, PRED_MEM, PRED_TIME).
set -uo pipefail
. "$(dirname "$0")/../env.sh"; . "$FEC_REPO/ch1/common.sh"
IN=${1:?usage: predict.sh <d3d_shot_dir> <run_name> [order] [out name]}; NAME=${2:?run name}; ORDER=${3:-sorted}; OUTNAME=${4:-}
RUN=${OUT_ROOT:-$SCR/stage11_ch1/final_path}/runs/$NAME
AFTER="echo '$CH1_PROFILE' > '$RUN/profile.txt'; bash '$FEC_REPO/ch1/check_heads.sh' '$RUN' '$CH1_PROFILE' '$HEADS'"
[ -z "$OUTNAME" ] || AFTER="$AFTER; cp -f '$RUN/submission.npz' '$RUN/$OUTNAME' && md5sum '$RUN/$OUTNAME' && sha256sum '$RUN/$OUTNAME' > '$RUN/$OUTNAME.sha256'"
# the profile's Development-phase md5 refers to the PUBLIC test: printed only when the output is named diii_d_public_test.npz
[ -z "$REF_MD5" ] || [ "$OUTNAME" != diii_d_public_test.npz ] || AFTER="$AFTER; echo \"profile $CH1_PROFILE: md5 \$(md5sum '$RUN/submission.npz' | cut -c1-32); Development-phase reference for diii_d_public_test: $REF_MD5\""
export AFTER
export YAUX_VAR=${YAUX_VAR:-$FEC_REPO/ch1/frozen/y_aux_var.json}
echo "=== ch1 predict profile=$CH1_PROFILE ($CH1_DESC) HEADS=$HEADS${HEAD22_ROOT:+ HEAD22_ROOT=$HEAD22_ROOT}"
INNER=$WS/stage11_run/ch1_package_final/predict.sh
# USE_SLURM=1 outside a job: the inner script submits itself (the package's own copy) and returns. One job of PRED_CPUS cores
# (default 72) with PRED_GPUS GPUs (default 1); memory default 100 GB per 72 cores (the allocation is billed by the largest of the
# core, GPU and memory shares of a node). PRED_GPUS=0 runs the job on the CPU (DEV=cpu).
if [ "${USE_SLURM:-0}" = 1 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  export PRED_MEM=${PRED_MEM:-$(( (${PRED_CPUS:-72} + 71) / 72 * 100 ))G}
  [ "${PRED_GPUS:-1}" = 0 ] && export DEV=${DEV:-cpu} CUDA_VISIBLE_DEVICES=
  exec bash "$INNER" "$IN" "$NAME" "$ORDER"
fi
# Device. DEV=cuda: FM / TW / SC / heads on the GPU (the devices of the shipped files); DEV=cpu: everything on the CPU (valid files,
# not bit-identical to the GPU ones; RUNBOOK_FINAL.md, CPU fallback). Without DEV the device is detected: a GPU visible to torch -> cuda,
# none -> cpu. DEV=cuda without a visible GPU stops here (the FM / TW stages would fail later).
GPU_OK=$("$WS/.venv/bin/python" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)
if [ -z "${DEV:-}" ]; then
  if [ "$GPU_OK" = 1 ]; then DEV=cuda; else DEV=cpu; fi
  echo "DEVICE: DEV not set -> DEV=$DEV ($([ "$GPU_OK" = 1 ] && echo "a GPU is visible" || echo "no GPU visible: CPU path, valid but not bit-identical to the GPU file"))"
elif [ "$DEV" = cuda ] && [ "$GPU_OK" != 1 ]; then
  echo "DEVICE: DEV=cuda but no GPU is visible to torch - stop (run on a node with a GPU, or set DEV=cpu for the CPU path)"; exit 2
else
  echo "DEVICE: DEV=$DEV (set in the environment)"
fi
export DEV
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=
bash "$INNER" "$IN" "$NAME" "$ORDER"; rc=$?
# A run is complete only if the writer AND the last hook ran. (A script file that is replaced while a job executes it - e.g. an
# update of the checkout on a network file system - ends the job silently with status 0 and without a file: seen in a rehearsal.)
if [ ! -s "$RUN/submission.npz" ] || [ ! -s "$RUN/profile.txt" ]; then
  echo "PREDICT_INCOMPLETE: $RUN has no finished submission (inner status $rc). Do not edit or replace files of the checkout while a run is in progress; run again with a new run name."
  exit 1
fi
exit $rc
