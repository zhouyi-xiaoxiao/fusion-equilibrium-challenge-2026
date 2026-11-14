#!/bin/bash
# TRAINING ENTRYPOINT of the MAST network that is trained on the inputs being predicted (the Final inputs in a Final run): the
# transductive network of the syn_*, int_* and dec_* profiles (recipe "XA", ch2/pipelines/syn_net.inc), which the submitted profile
# dec_e4s_c1nx (ch2/pipelines/DEFAULT) contains. GPU; its GPU time counts toward the GPU-hour limit of the rules (PROVENANCE.md
# section 4). The network of each Final submission is the one trained by these same predict_mast.sh calls in that submission's
# Final run (there: the init / syngen / synreal / syntrain jobs of tools/final_slurm.sh; this wrapper was added after the Final checkout).
#
#   [PIPE=<profile>] [NSHARD=32] [SHARD=k] [STEPS="..."] [DEVICE=gpu|cpu] [DRYRUN=1] \
#       bash ch2/train_mast_net.sh <mast_shot_dir> <run_name> [<output file name>]
#
# Only orchestration: every step is one call of the unchanged ch2/predict_mast.sh with the same arguments as the jobs of
# tools/final_slurm.sh (steps init, syngen, synreal, syntrain; RUNBOOK_FINAL.md B.3 / B.4), one after the other on this machine.
# Run directory: $SCR/ch2_runs/<run_name>, SCR = FEC_WORK unless SCR is set (env.sh; the same directory the inference stages of that
# run use); <output file name>, default mast_private_test.npz, is only passed on - none of these stages writes it.
# Steps (STEPS, default "init syngen synreal syntrain"):
#   init      run directory, sorted shot list and the NSHARD shard lists (predict_mast.sh STAGES=init); once, never with SHARD
#   syngen    synthetic free-boundary equilibria of our own forward model, driven by the input columns of these shots (CPU; 333 core-h
#             for 1,206 shots). One call per shard (SHARD=k, as the jobs of final_slurm.sh), k = 0 .. NSHARD-1; finished shards are
#             skipped, an interrupted shard resumes. With SHARD=k set here: shard k only (one process per machine, any order; default
#             STEPS="syngen", and the run directory must have been initialised once with STEPS=init); then call again without SHARD.
#   synreal   network input features of every frame of these shots (CPU, seconds)
#   syntrain  training of the network on the synthetic set (1 GPU, about 3 min on one GH200) and its prediction of every frame
# At the end: the model file <run dir>/syn/models/model_<SYN_NAME>.pt, its sha256, the training device and the training wall time
# from the trainer's log line "saved ...; wall <n>s" (<run dir>/logs/syntrain.log). The generation is seeded per shot, so the
# result does not depend on NSHARD; NSHARD must equal the number of shard lists of an existing run directory (checked).
# DEVICE=gpu (default, as final_slurm.sh): stop unless torch sees a GPU. DEVICE=cpu: syntrain on the CPU (CPU fallback; a valid
# network, not bit-identical to the GPU one). SYN_MODEL_FILE must not be set: with it predict_mast.sh copies a given network
# instead of training (that is the inference path with a published network, README.md "Quick start").
# Needs what predict_mast.sh checks on every call: the environments (env/build_env.sh) and, for dec_* / int_* profiles, the files of
# weights/MANIFEST_CH2.sha256 in FEC_WORK (tools/fetch_weights.sh), although the network steps do not read them.
# DRYRUN=1 prints the commands without running them.
# The calls get the environment of the jobs of tools/final_slurm.sh (sbatch --export=NONE): the caller's PYTHONPATH / VIRTUAL_ENV and
# the overrides of geometry, constants, model paths and network threads (GEOM, CONST, EXTRA_CONST, LI_MODEL, CORR_MODEL, DEC_CFG,
# DEC_SUPSRC, SYN_THREADS) are removed; NPROC (worker processes; the synthetic set is seeded per shot) is kept.
# On a cluster, run init / syngen / synreal in CPU jobs (STEPS=..., SHARD=k per syngen job) and only syntrain in a job with 1 GPU,
# as tools/final_slurm.sh does: the counted GPU time is the elapsed time of that one job (Final-input network about 0.07 GPU-h).
# Inference after this step: the solve shards and the finish stages of predict_mast.sh (merge scal pack post validate, with the
# stream-order file), RUNBOOK_FINAL.md B.3 / B.5, with the SAME NSHARD (predict_mast.sh itself defaults to NSHARD=1, which on a
# 32-shard run directory solves shard 0 only and stops at merge).
set -uo pipefail
unset PYTHONPATH VIRTUAL_ENV GEOM CONST EXTRA_CONST LI_MODEL CORR_MODEL DEC_CFG DEC_SUPSRC SYN_THREADS   # as the jobs of tools/final_slurm.sh
FEC_REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$FEC_REPO/env.sh" || exit 1                   # WS, SCR: the values predict_mast.sh resolves from the same environment
usage="usage: [PIPE=<profile>] [NSHARD=32] [SHARD=k] [STEPS=...] [DEVICE=gpu|cpu] [DRYRUN=1] bash ch2/train_mast_net.sh <mast_shot_dir> <run_name> [<output file name>]"
[ $# -ge 2 ] || { echo "$usage"; exit 2; }
MD=$(CDPATH= cd -- "$1" 2>/dev/null && pwd -P) || { echo "not a directory: $1"; exit 2; }
NAME=$2; OUTNAME=${3:-mast_private_test.npz}
PIPE=${PIPE:-$(tr -d ' \n' < "$FEC_REPO/ch2/pipelines/DEFAULT")}
NSHARD=${NSHARD:-32}; DEVICE=${DEVICE:-gpu}; DRY=${DRYRUN:-0}; K=${SHARD:-}
O=$SCR/ch2_runs/$NAME
case $DEVICE in gpu|cpu) ;; *) echo "DEVICE=gpu|cpu"; exit 2;; esac
[[ "$NSHARD" =~ ^[1-9][0-9]*$ ]] || { echo "NSHARD must be a positive integer"; exit 2; }
if [ -n "$K" ]; then
  [[ "$K" =~ ^(0|[1-9][0-9]*)$ ]] && [ "$K" -lt "$NSHARD" ] || { echo "SHARD must be an integer in 0 .. $((NSHARD - 1)) (no leading zeros)"; exit 2; }
  STEPS=${STEPS:-syngen}
else
  STEPS=${STEPS:-init syngen synreal syntrain}
fi
for s in $STEPS; do
  case $s in
    init) [ -z "$K" ] || { echo "init runs once, without SHARD, before the shard processes"; exit 2; };;
    syngen) ;;
    synreal|syntrain) [ -z "$K" ] || { echo "with SHARD=$K only the step syngen runs (synreal and syntrain need every shard): call again without SHARD"; exit 2; };;
    *) echo "unknown step $s (steps: init syngen synreal syntrain)"; exit 2;;
  esac
done
has() { [[ " $STEPS " == *" $1 "* ]]; }
[ -z "${SYN_MODEL_FILE:-}" ] || { echo "SYN_MODEL_FILE is set: that copies a given network instead of training it (inference path); unset it to train"; exit 2; }

# the profile's network settings, read in a subshell as tools/final_slurm.sh does (the profile files are only sourced, not changed)
Q=$( set +u; export PIPE; . "$FEC_REPO/ch2/common.sh" >/dev/null 2>&1 || exit 2; echo "${PSTAGES[*]:-}|${SYN_NAME:-}|${RUN_MODELS[*]:-}" )
[ -n "$Q" ] || { echo "unknown or invalid MAST profile '$PIPE' (rejected by ch2/common.sh: no ch2/pipelines/$PIPE.sh, or it does not set TAG / SOLVER_DIR / SOLVER)"; exit 2; }
PST=${Q%%|*}; R=${Q#*|}; SYN_NAME=${R%%|*}; RUN_MODEL=${R#*|}
for s in syngen synreal syntrain; do
  [[ " $PST " == *" $s "* ]] || { echo "MAST profile $PIPE has no network stages (profile stages: '${PST}'): nothing to train"; exit 2; }
done
[ -n "$SYN_NAME" ] && [ -n "$RUN_MODEL" ] || { echo "MAST profile $PIPE: SYN_NAME / RUN_MODELS not set"; exit 2; }
M=$O/$RUN_MODEL
PYS=$WS/.venv/bin/python

# an existing run directory keeps its shard split (merge and the shard summaries refer to it)
if [ -d "$O/shards" ]; then
  n=$(ls "$O/shards" | grep -c '^list_' || true)
  [ "$n" = 0 ] || [ "$n" = "$NSHARD" ] || { echo "run directory $O has $n shard lists: set NSHARD=$n"; exit 2; }
fi
if [ -n "$K" ] && [ "$DRY" != 1 ] && [ ! -s "$O/shards/list_$K.txt" ]; then
  echo "run directory $O is not initialised: run STEPS=init once (without SHARD) before the shard processes"; exit 2
fi

echo "=== ch2 train_mast_net $(hostname) $(date -u) dir=$MD name=$NAME pipeline=$PIPE nshard=$NSHARD shard=${K:-all} steps=$STEPS device=$DEVICE run_dir=$O"
if has syntrain && [ "$DEVICE" = gpu ] && [ "$DRY" != 1 ]; then
  [ "$("$PYS" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null)" = 1 ] || { echo "NO_GPU_VISIBLE (DEVICE=gpu): stop; DEVICE=cpu for the CPU fallback"; exit 1; }
fi

T0=$(date +%s); TS=0
pm() {  # <stage> [<shard>]: one call of the inference entrypoint, environment as in the jobs of tools/final_slurm.sh
  local st=$1 k=${2:-} t; local e=(PIPE="$PIPE" NSHARD="$NSHARD" STAGES="$st")
  [ -n "$k" ] && e+=(SHARD="$k")
  [ "$st" = syntrain ] && [ "$DEVICE" = cpu ] && e+=(CUDA_VISIBLE_DEVICES=)
  if [ "$DRY" = 1 ]; then      # printed shell-quoted, so that it can be pasted
    printf 'cd %q && env -u SHARD -u SYN_MODEL_FILE' "$FEC_REPO"; printf ' %q' "${e[@]}" bash ch2/predict_mast.sh "$MD" "$NAME" sorted "$OUTNAME"; echo
    return 0
  fi
  t=$(date +%s)
  ( cd "$FEC_REPO" && env -u SHARD -u SYN_MODEL_FILE "${e[@]}" bash ch2/predict_mast.sh "$MD" "$NAME" sorted "$OUTNAME" ) || return 1
  TS=$(( $(date +%s) - t ))
}

if has init; then
  pm init || { echo "TRAIN_STEP_FAILED init"; exit 1; }
  if [ "$DRY" != 1 ]; then
    n=$(ls "$O/shards" 2>/dev/null | grep -c '^list_' || true)
    [ "$n" = "$NSHARD" ] || { echo "TRAIN_STEP_FAILED init: $n shard lists, expected $NSHARD"; exit 1; }
  fi
fi
if has syngen; then
  if [ -n "$K" ]; then ks=$K; else ks=$(seq 0 $((NSHARD - 1))); fi
  for k in $ks; do
    pm syngen "$k" || { echo "TRAIN_STEP_FAILED syngen shard $k: run the same command again (finished shards are skipped)"; exit 1; }
  done
  [ -n "$K" ] && [ "$DRY" != 1 ] && echo "syngen shard $K of $NSHARD done; when every shard is done: the same command without SHARD (STEPS=\"synreal syntrain\")"
fi
has synreal && { pm synreal || { echo "TRAIN_STEP_FAILED synreal"; exit 1; }; }
if has syntrain; then
  PRE=0; [ -s "$M" ] && PRE=1
  pm syntrain || { echo "TRAIN_STEP_FAILED syntrain"; exit 1; }
  if [ "$DRY" = 1 ]; then
    echo "then: model $M, its sha256, training device and wall time ($O/logs/syntrain.log, $O/syn/models/train_$SYN_NAME.json)"
    echo "DRYRUN: nothing was run"; exit 0
  fi
  [ -s "$M" ] || { echo "TRAIN_STEP_FAILED syntrain: no model file $M"; exit 1; }
  if command -v sha256sum >/dev/null 2>&1; then SHA=$(sha256sum "$M" | cut -c1-64); else SHA=$(shasum -a 256 "$M" | cut -c1-64); fi
  WL=$(grep -E "saved .*model_$SYN_NAME\.pt; wall [0-9]+s" "$O/logs/syntrain.log" 2>/dev/null | tail -1)
  WALL=$(echo "$WL" | sed -nE 's/.*; wall ([0-9]+)s.*/\1/p')
  DEV=$("$PYS" -c "import json,sys; print(json.load(open(sys.argv[1])).get('device', '?'))" "$O/syn/models/train_$SYN_NAME.json" 2>/dev/null)
  echo "TRAIN_MAST_NET model $M"
  echo "TRAIN_MAST_NET sha256 $SHA"
  echo "TRAIN_MAST_NET training wall_s ${WALL:-?} device ${DEV:-?} (trainer log: ${WL:-no 'saved ... wall' line in $O/logs/syntrain.log})"
  if [ $PRE = 1 ]; then
    echo "TRAIN_MAST_NET note: the model file existed before this call, so this call did not train it (training figures = the earlier call's log)"
  elif [ "${DEV:-}" = cuda ]; then
    echo "TRAIN_MAST_NET syntrain stage ${TS} s on 1 GPU = $(awk -v t="$TS" 'BEGIN { printf "%.3f", t / 3600 }') GPU-h (training + prediction of every frame)"
  elif [ "${DEV:-}" = cpu ]; then
    echo "TRAIN_MAST_NET syntrain stage ${TS} s on the CPU (training + prediction of every frame; no GPU time)"
  else
    echo "TRAIN_MAST_NET syntrain stage ${TS} s, device unknown (no readable $O/syn/models/train_$SYN_NAME.json): count it as GPU time if the job had a GPU"
  fi
  printf 'TRAIN_MAST_NET next (inference on this run directory, the same NSHARD): PIPE=%q NSHARD=%q STAGES=solve [SHARD=k] bash ch2/predict_mast.sh %q %q sorted %q; then PIPE=%q NSHARD=%q STAGES="merge scal pack post validate" bash ch2/predict_mast.sh %q %q <stream_order.txt> %q\n' \
      "$PIPE" "$NSHARD" "$MD" "$NAME" "$OUTNAME" "$PIPE" "$NSHARD" "$MD" "$NAME" "$OUTNAME"
fi
[ "$DRY" = 1 ] && { echo "DRYRUN: nothing was run"; exit 0; }
echo "TRAIN_MAST_NET_DONE steps=$STEPS wall_s $(( $(date +%s) - T0 ))"
