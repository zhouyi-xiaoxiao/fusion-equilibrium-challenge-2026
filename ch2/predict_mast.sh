#!/bin/bash
# Ch2 inference entrypoint: MAST input shots (parquet, inputs only) -> submission .npz. CPU only, label-free.
#
#   [PIPE=<profile>] bash ch2/predict_mast.sh <mast_shot_dir> <run_name> [<stream_order.txt>|sorted] [<output file name>]
#
# PIPE selects the pipeline profile ch2/pipelines/<PIPE>.sh (default: ch2/pipelines/DEFAULT). Stages (STAGES="solve merge scal pack
# post validate" by default):
#   solve     the profile's solver per shot (frames of a shot are solved in sequence, shots are independent). NSHARD=N splits the
#             sorted shot list into N interleaved shards; SHARD=k solves shard k only (one process per machine/node, any order);
#             without SHARD all shards are solved one after another on this machine. A shard is complete when every listed shot has
#             its part file and no shot ended in a fatal error; re-running the same command resumes an incomplete shard (finished
#             shots are kept, the others are solved again).
#   merge     collect the shard outputs into one run directory in global (sorted) shot order
#   scal      beta_N table from the Thomson pressure mapped on the solved flux (ch2d_scalars/scal_frames.py)
#   pack      q95 = contour integral x C_q95, beta_N = c x beta_N,e (DIII-D-derived constants in ch2/constants/<PIPE>), validity
#             window, 3-frame median, float32, written in sorted file-name order (the profile's packer); if a stream-order file is
#             given and differs from the sorted order, the final file is put in stream order (tools/reorder_npz.py)
#   <name>    profile stages (PSTAGES of the profile, functions stage_<name> <run dir> <shot dir>), run after solve in the profile's
#             order, e.g. the transductive network stages of the syn_* profiles (synreal: network input features; syngen: synthetic
#             equilibria driven by the input columns of the shots being predicted, shardable with NSHARD / SHARD like solve;
#             syntrain: GPU training of the network on them + its prediction of every frame). Each stage resumes / skips finished work.
#   post      the profile's post-processing stages, in order (none for p7a)
#   validate  local format check (shot i of the file against the input file at stream index i of the given order file; sorted file
#             names when the order is "sorted"); VAL_CONFIG=<HF config> (+ VAL_REPO, default the Development dataset) also runs the
#             organisers' validator (unmodified, through tools/validate_final.py, which also knows the Final configurations)
# Outputs: $SCR/ch2_runs/<run_name>/submission/<output file name> (default mast_public_test.npz) + md5.txt, sha256.txt, record.json
# Every process that reads a MAST file installs rule3_guard (input-column allow-list; refuses label / envelope / demo paths).
set -uo pipefail
. "$(dirname "$0")/../env.sh"; . "$FEC_REPO/ch2/common.sh"
MD=$(readlink -f "${1:?usage: predict_mast.sh <mast_shot_dir> <run_name> [order] [out name]}"); NAME=${2:?run name}; ORDER=${3:-sorted}; OUTNAME=${4:-mast_public_test.npz}
[ "$ORDER" = sorted ] || ORDER=$(readlink -f "$ORDER")
O=$SCR/ch2_runs/$NAME; NSHARD=${NSHARD:-1}; STAGES=${STAGES:-solve ${PSTAGES[*]} merge scal pack post validate}
has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$O/shards" "$O/logs"; T0=$(date +%s); el() { echo "--- [$(( $(date +%s) - T0 ))s] $*"; }
echo "=== ch2 predict $(hostname) $(date -u) dir=$MD name=$NAME pipeline=$PIPE tag=$TAG nshard=$NSHARD shard=${SHARD:-all} cores=$NPROC stages=$STAGES"
[ -s "$CONST/calib_$TAG.json" ] && [ -s "$CONST/c_${TAG}_mast.json" ] && [ -s "$GEOM/mast_limiter_mask.npz" ] || { echo "missing constants ($CONST) or geometry"; exit 2; }
for m in "${MODELS[@]}"; do [ -s "$m" ] || { echo "missing model file $m (tools/fetch_weights.sh)"; exit 2; }; done
# one run directory belongs to one pipeline
if [ -s "$O/pipeline.txt" ]; then [ "$(cat "$O/pipeline.txt")" = "$PIPE" ] || { echo "run $O was started with pipeline $(cat "$O/pipeline.txt"), not $PIPE"; exit 2; }; else echo "$PIPE" > "$O/pipeline.txt"; fi
# shard lists (deterministic: sorted names, interleaved)
if [ ! -s "$O/shards/all.txt" ]; then ( cd "$MD" && ls *.parquet | LC_ALL=C sort ) > "$O/shards/all.txt"; fi
for k in $(seq 0 $((NSHARD - 1))); do [ -s "$O/shards/list_$k.txt" ] || awk -v n=$NSHARD -v k=$k '(NR - 1) % n == k' "$O/shards/all.txt" > "$O/shards/list_$k.txt"; done
solve_one() {
  local k=$1 S=$O/shards/s$1 L=$O/shards/list_$1.txt
  if shard_ok "$S" "$L"; then echo "shard $k already solved"; return 0; fi
  if [ -f "$S/summary_${TAG}_mast.json" ]; then mv -f "$S/summary_${TAG}_mast.json" "$S/summary_${TAG}_mast.json.incomplete_$(date -u +%H%M%S)"; echo "shard $k incomplete: resuming"; fi
  rm -f "$S"/parts/*.tmp.npz
  # the shard's input directory: links to its shots (every solver takes --data-dir; shot indices are local to the shard)
  local IN=$O/shards/in_$k
  if [ "$(ls "$IN" 2>/dev/null | wc -l)" != "$(grep -c . "$L")" ]; then rm -rf "$IN"; mkdir -p "$IN"; while read -r n; do [ -n "$n" ] && ln -s "$MD/$n" "$IN/$n"; done < "$L"; fi
  run_solver --machine mast --mode real --data-dir "$IN" --wall "$GEOM/mast_limiter_mask.npz" "${MAST_ARGS[@]}" \
      --tag "$TAG" --out "$S" --procs "$NPROC" >> "$O/logs/solve_$k.log" 2>&1
  grep -E "SUMMARY|Traceback|Error" "$O/logs/solve_$k.log" | tail -3 | cut -c1-600
  shard_ok "$S" "$L" || { echo "SOLVE_INCOMPLETE shard $k: run the same command again (finished shots are kept); see $O/logs/solve_$k.log"; return 1; }
}
if has solve; then
  if [ -n "${SHARD:-}" ]; then solve_one "$SHARD" || exit 1; else for k in $(seq 0 $((NSHARD - 1))); do solve_one "$k" || exit 1; done; fi
  el solve
fi
for p in "${PSTAGES[@]}"; do
  if has "$p"; then "stage_$p" "$O" "$MD" || { echo "STAGE_FAILED $p"; exit 1; }; el "$p"; fi
done
if has merge; then
  for k in $(seq 0 $((NSHARD - 1))); do shard_ok "$O/shards/s$k" "$O/shards/list_$k.txt" || { echo "MERGE_REFUSED: shard $k is incomplete"; exit 1; }; done
  $PYS "$FEC_REPO/ch2/merge_shards.py" --run "$O" --tag "$TAG" --machine mast --nshard "$NSHARD" || { echo MERGE_FAILED; exit 1; }
  el merge
fi
if has scal; then
  $PY "$SCALC/scal_frames.py" --machine mast --src g1 --run "$O" --tag "$TAG" --data-dir "$MD" --wall "$GEOM/mast_limiter_mask.npz" \
      --out "$O/scal_mast_$TAG.parquet" --procs "$NPROC" 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-600
  [ -s "$O/scal_mast_$TAG.parquet" ] || { echo SCAL_FAILED; exit 1; }
  el scal
fi
S=$O/submission; mkdir -p "$S" "$O/post"
REORDER=0; [ "$ORDER" != sorted ] && ! cmp -s <(grep . "$ORDER") <(grep . "$O/shards/all.txt") && REORDER=1
FINAL=$S/$OUTNAME; [ $REORDER = 1 ] && FINAL=$O/post/sorted_$OUTNAME     # written in sorted order, then put in stream order
PACKED=$FINAL; [ ${#POST[@]} -eq 0 ] || PACKED=$O/post/packed_$OUTNAME
if has pack; then
  CQ=$($PYS -c "import json; print(repr(json.load(open('$CONST/calib_$TAG.json'))['C_q95']))")
  echo "C_q95 = $CQ ; c_betaN = $($PYS -c "import json; print(repr(json.load(open('$CONST/c_${TAG}_mast.json'))['c_used']))")"
  # packed in sorted file-name order; a different stream order is applied after the post stages (tools/reorder_npz.py)
  [ -f "$PACKED" ] && { echo "refusing to overwrite $PACKED"; exit 3; }
  $PY "$PACKER" --run "$O" --tag "$TAG" --machine mast --scal "$O/scal_mast_$TAG.parquet" --c-json "$CONST/c_${TAG}_mast.json" \
      --psi-bits 24 --q-factor "$CQ" --out "$PACKED" | cut -c1-1500
  [ -s "$PACKED" ] || { echo PACK_FAILED; exit 1; }
  el pack
fi
if has post && [ ${#POST[@]} -gt 0 ]; then
  cur=$PACKED; i=0
  for p in "${POST[@]}"; do
    i=$((i + 1)); nxt=$O/post/${i}_${p}_$OUTNAME; [ $i -eq ${#POST[@]} ] && nxt=$FINAL
    [ -f "$nxt" ] && { echo "refusing to overwrite $nxt"; exit 3; }
    "post_$p" "$O" "$cur" "$nxt" "$MD" || { echo "POST_FAILED $p"; exit 1; }
    echo "post $p -> $(md5sum "$nxt" | cut -c1-32)"; cur=$nxt
  done
  el post
fi
if [ $REORDER = 1 ] && [ -s "$FINAL" ] && [ ! -f "$S/$OUTNAME" ] && { has pack || has post; }; then
  $PYS "$FEC_REPO/tools/reorder_npz.py" "$FINAL" "$O/shards/all.txt" "$ORDER" "$S/$OUTNAME" || { echo REORDER_FAILED; exit 1; }
fi
if [ -s "$S/$OUTNAME" ] && { has pack || has post; }; then
  ( cd "$S" && md5sum "$OUTNAME" | tee md5.txt && sha256sum "$OUTNAME" > sha256.txt )
  pipeline_record "$S/record.json" "$O"
fi
if has validate; then
  VO=(); [ "$ORDER" = sorted ] || VO=(--order "$ORDER")       # the file is in stream order: check shot i against the input at stream index i
  $PYS "$WS/stage6_run/variants/P_ops_mast_package/validate_local.py" "$S/$OUTNAME" --shot-dir "$MD" ${VO[@]+"${VO[@]}"} > "$S/validate_local.log" 2>&1; echo "validate_local rc=$? | $(tail -1 "$S/validate_local.log" | cut -c1-200)"
  if [ -n "${VAL_CONFIG:-}" ]; then
    ( cd "$S" && $PYS "$FEC_REPO/tools/validate_final.py" "$OUTNAME" --repo "${VAL_REPO:-Sophelio/fusion-equilibrium-challenge}" --config "$VAL_CONFIG" --max-shots 0 > validate_official.log 2>&1; echo "official validator rc=$? | $(grep -v '^\s*$' validate_official.log | tail -1 | cut -c1-200)" )
  fi
  el validate
fi
# the profile's Development-phase md5 refers to the PUBLIC test; for any other output name (e.g. the Final's) there is no reference
[ -n "$REF_MD5" ] && [ "$OUTNAME" = mast_public_test.npz ] && [ -s "$S/md5.txt" ] && echo "pipeline $PIPE: md5 $(cut -c1-32 "$S/md5.txt"); Development-phase reference for mast_public_test: $REF_MD5"
echo "CH2_PREDICT_DONE $S/$OUTNAME wall_s $(( $(date +%s) - T0 ))"
