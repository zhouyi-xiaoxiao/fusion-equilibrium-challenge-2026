#!/bin/bash
# Ch1 training entrypoint of the DIII-D profile xhc: everything the predict path of profile xhc loads, from scratch, on the 7,041
# labelled DIII-D shots (fixed epochs and seeds, no validation, no early stopping).
#
#   [USE_SLURM=1] [CMP=1] bash ch1/train_xhc.sh <MODEL_ROOT> <diii_d_train dir>
#
#   1  base + heads  = ch1/train.sh with CH1_PROFILE=xhc (unchanged; identical to profile hc10x24): FM 11 + TW 4 + SC 17 on the
#                      canonical v2.2 cache -> MODEL_ROOT/base; 24 corrector heads (v2.2 c10) -> MODEL_ROOT/heads22
#   2  FT            = ch1/train_xhc_ft.sh MODEL_ROOT/base MODEL_ROOT <diii_d_train dir>: label-polygon contour fine-tune of the 11 FM
#                      members and the 4 TW seeds of MODEL_ROOT/base -> MODEL_ROOT/xhc (the base root of the predict path)
# USE_SLURM=1: stage 1 submits its jobs (base 4 GPUs, heads 4 x 1 GPU), stage 2 waits for the base job (AFTER). Local: one after the other.
# CMP=1: stage 1 writes MODEL_ROOT/wcmp.json (base + heads vs weights/MANIFEST*.sha256 files), stage 2 MODEL_ROOT/xwcmp.json (vs
# weights/MANIFEST_CH1_XHC.sha256). Prerequisites: generated data (weights/GENERATED.sha256, weights/GENERATED_XHC.sha256) and the
# public diii_d_train parquet files.
# Then:  CH1_PROFILE=xhc BASE_ROOT=MODEL_ROOT/xhc HEAD_ROOT=MODEL_ROOT/heads HEAD22_ROOT=MODEL_ROOT/heads22 bash ch1/predict.sh <shot_dir> <name>
# GPU time (development, GH200): base 1.65 + heads 1.1 (+ FcCX-style heads job cancelled at once) + FT 1.9 = about 4.7 GPU-h.
set -euo pipefail
ROOT=${1:?usage: train_xhc.sh <MODEL_ROOT> <diii_d_train dir>}; DTR=${2:?diii_d_train dir}
HERE=$(cd "$(dirname "$0")" && pwd)
export CH1_PROFILE=xhc
mkdir -p "$ROOT"; ROOT=$(readlink -f "$ROOT")
if [ "${USE_SLURM:-0}" = 1 ]; then
  bash "$HERE/train.sh" "$ROOT"
  JB=$(awk '{print $1}' "$ROOT/train_jobs.txt")
  [ -n "$JB" ] || { echo "no base job id in $ROOT/train_jobs.txt"; exit 1; }
  AFTER=$JB bash "$HERE/train_xhc_ft.sh" "$ROOT/base" "$ROOT" "$DTR"
  echo "TRAIN_XHC_SUBMITTED root=$ROOT base job $JB; stage-1 jobs $(cat "$ROOT/train_jobs.txt"); FT jobs $(cat "$ROOT/xft_jobs.txt")"
else
  bash "$HERE/train.sh" "$ROOT"
  bash "$HERE/train_xhc_ft.sh" "$ROOT/base" "$ROOT" "$DTR"
  echo "TRAIN_XHC_DONE root=$ROOT"
fi
