#!/bin/bash
# ch1_package_final — Ch1 FINAL predict path for the FcCX recipe. ONE command, ANY directory of DIII-D parquet shots (inputs only):
#
#   bash predict.sh <shot_dir> <run_name> [stream_order.txt|sorted]
#
# Runs inline on the current machine (all cores, one GPU). With USE_SLURM=1 it submits itself as ONE job (PRED_GPUS=1 PRED_CPUS=72 PRED_MEM=200G PRED_TIME=04:00:00 by default;
# PRED_GPUS=4 PRED_CPUS=288 PRED_MEM=440G uses a whole GH200 node: only 1 GPU is used, the CPU steps scale with the cores).
# Recipe "FcCX" (see METHODS.md). No label column is read;
# efit_* / magnetics_dsep are never inputs (efit_times / efit_grid_R / efit_grid_Z are the query inputs).
#   1 features   pf_features.py   starter data_fixes + FEATURE-V2.1 rows (heads) and FEATURE-V2.2 rows (base: guarded 8192 ms old-DAQ
#                                 Ip-clock rule with its own fallback), per-shot sign class / sign and clock record (never aborts)
#   2 canonical  pf_canon.py      430 odd columns x explicit per-shot sign, raw sign flag kept (cfl), then the canonical TRAIN scalers:
#                                 base = v2.2 rows with $BASE_SCALER, heads = v2.1 rows with $HEAD_SCALER
#   3 FM         pf_fm.py         11 psi nets (5 d30aux + 3 tower + 3 K120), GPU fp32
#   4 TW         ch1_temporal/predict_unseen.py   tR_w9d seeds 0-3, GPU fp32
#   5 SC         ch1_scalars/sc_predict.py        bl_d10, 17 members / 7 families, GPU
#   6 blend      pf_blend.py      psi = (FM + TW)/2 ; q95/betaN = SC
#   7 corrector  ch1_corrector/run_correct.sh     Jacobian of the base (CPU, ~70 core-h per 193k frames), canonical direct2 heads
#                                 (6 seeds, GPU bf16), frozen canonical rob constants, KEEP_BASE=volume, incumbent GN r1e+09_e1_i13
#   8 robmi      ch1d_specialists/sp_apply.py     metric-prior GN on the same Jacobian and rob targets (7 targets, M_80, trace ridge)
#   9 final step FINAL_STEP=gM (default) = the VALIDATED fold chain first: robmi -> li window guard (ch1d_liloss/li_guard.py) ->
#                                 census with the scorer's own derivation; only if a frame still fails: take the incumbent-GN frame
#                                 where that derives finitely (sp_merge.py) and guard again.   FINAL_STEP=Mg = merge first, then guard.
#                                 Zero derivation failures is a hard gate (G6): a residual failure is reported, the file is still written.
#  10 writer     write_submission.py              float32 psi / q95 / betaN in the given order, never rounded
#  11 gate       f_ch1_harden/final_gate.py --clock v22 (G1-G10) ; VAL_CONFIG=<HF config> also runs the starter validator
# Device policy = the devices that produced the shipped file: FM / TW / SC / heads on the GPU (DEV=cpu runs FM and TW on the CPU; valid,
# not bit-identical).  DEVCHECK=1 additionally writes CPU FM / TW files for comparison only.
# Weights / frozen artefacts (override to use a train.sh root):
#   BASE_ROOT    FM / TW / SC FULL weights            [$S11/e_cx/full]
#   BASE_FEAT    canonical v2.2 cache: scaler_v21.npz, spec.json, names.txt   [$S11/e_cx/h150/feat]
#   HEAD_ROOT    canonical FULL direct2 heads ck_direct2_s{0..5}.pt + blend_frozen_gv21c.json   [$S11/f_corr_canon/full/corrector]
#   HEADS        v22c10 (default here; ch1/predict.sh sets it from the selected profile) = direct heads on the canonical FEATURE-V2.2 rows (the base's own cache FB), width 1024,
#                8 blocks, dropout 0.15, 24 seeds (tags c10a..d x 6), predicted with stage11_run/l_corr/l_hp.py into hp_direct2_mlp_pkg.npy
#                before the corrector;  v21 = the FcCX heads (ck_direct2_s0-5 on the v2.1 rows) exactly as before.
#   HEAD22_ROOT  FULL v2.2 heads ck_c10{a,b,c,d}_s*.pt   [$S11/l_corr/heads22/full]
#   HEAD_SCALER  canonical v2.1 scaler                [$S11/e_fold2_offset/h150_cfl/feat/scaler_v21.npz]
#   OOF_ROOT     gv21 OOF statistics cov_oof_gv21.npy steps_gv21.npy dsc_gv21.json   [$S11/corrector_full]
#   YAUX_VAR     the 11 training variances of the scalar targets used by the robmi trace ridge (frozen statistics, json)   [$S11/feat_v21_all/train/y_aux_var.json]
#   METRIC80     [$S11/metric/M_80.npy]      PCA_DIR  [$SCR/stage6_H]
# Outputs: $OUT_ROOT/runs/<run_name>/  (OUT_ROOT default $S11/final_path) ; FORCE=1 reuses a run name.
set -uo pipefail
WS=${WS:?set WS}
SCR=${SCR:?set SCR}
S11=$SCR/stage11_ch1
PK=$WS/stage11_run/ch1_package_final
OUT_ROOT=${OUT_ROOT:-$S11/final_path}
IN_DIR=${1:?usage: predict.sh <shot_dir> <run_name> [order]}; NAME=${2:?run name}; ORDER=${3:-sorted}
IN_DIR=$(readlink -f "$IN_DIR")
[ "$ORDER" = sorted ] || ORDER=$(readlink -f "$ORDER")
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${USE_SLURM:-0}" = 1 ]; then
  SELF=$(readlink -f "${BASH_SOURCE[0]}")       # the job runs THIS script; HEADS and the other settings travel with --export=ALL
  export HEADS=${HEADS:-v22c10}
  mkdir -p "$OUT_ROOT/logs"
  exec sbatch -J "pf_$NAME" -p ${SLURM_PARTITION:-workq} -N 1 --gpus=${PRED_GPUS:-1} --cpus-per-task=${PRED_CPUS:-72} --mem=${PRED_MEM:-200G} -t "${PRED_TIME:-04:00:00}" \
    -o "$OUT_ROOT/logs/slurm_predict_${NAME}_%j.out" --export=ALL "$SELF" "$IN_DIR" "$NAME" "$ORDER"
fi
{ [ -d "${TMPDIR:-/nonexistent}" ] && [ -w "$TMPDIR" ]; } || export TMPDIR=/tmp
BASE_ROOT=${BASE_ROOT:-$S11/e_cx/full}; BASE_FEAT=${BASE_FEAT:-$S11/e_cx/h150/feat}
HEAD_ROOT=${HEAD_ROOT:-$S11/f_corr_canon/full/corrector}; HEAD_SCALER=${HEAD_SCALER:-$S11/e_fold2_offset/h150_cfl/feat/scaler_v21.npz}
HEADS=${HEADS:-v22c10}; HEAD22_ROOT=${HEAD22_ROOT:-$S11/l_corr/heads22/full}; LC=$WS/stage11_run/l_corr
OOF_ROOT=${OOF_ROOT:-$S11/corrector_full}; YAUX_VAR=${YAUX_VAR:-$S11/feat_v21_all/train/y_aux_var.json}
M80=${METRIC80:-$S11/metric/M_80.npy}; export PCA_DIR=${PCA_DIR:-$SCR/stage6_H}
FINAL_STEP=${FINAL_STEP:-gM}; DEV=${DEV:-cuda}
SPLIT=pkg; TAG=pk
RUN=$OUT_ROOT/runs/$NAME
if [ -e "$RUN" ]; then [ "${FORCE:-0}" = 1 ] || { echo "run dir exists: $RUN (FORCE=1 to redo)"; exit 1; }; rm -rf "$RUN"; fi
FEAT=$RUN/feat; FB=$RUN/feat_base; FH=$RUN/ccfeat/$SPLIT; CCO=$RUN/cc
mkdir -p "$RUN" "$RUN/ccfeat/train" "$CCO/logs" "$CCO/tmp"
source "$WS/.venv/bin/activate"
export WS SCR PYTHONUNBUFFERED=1 XDG_CACHE_HOME=$S11/xdg_cache MPLCONFIGDIR=$S11/.mplcache CUBLAS_WORKSPACE_CONFIG=:4096:8
NC=${SLURM_CPUS_PER_TASK:-$(nproc)}; NW=$(( NC > 16 ? NC - 8 : NC )); TH=$(( NC > 64 ? 64 : NC ))
T0=$(date +%s); el() { echo "--- [$(( $(date +%s) - T0 ))s] $*"; }
die() { echo "PREDICT_FAILED at $*"; exit 1; }
echo "=== predict ${SLURM_JOB_ID:-local} on $(hostname) $(date -u) IN_DIR=$IN_DIR NAME=$NAME ORDER=$ORDER FINAL_STEP=$FINAL_STEP DEV=$DEV cores=$NC workers=$NW HEADS=$HEADS"
echo "n_parquet $(ls "$IN_DIR"/*.parquet | wc -l) | gpu: $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
H0=""; [ "$HEADS" = v21 ] && H0="$HEAD_ROOT/ck_direct2_s0.pt"
for f in "$BASE_FEAT/scaler_v21.npz" "$BASE_FEAT/spec.json" "$HEAD_SCALER" $H0 "$HEAD_ROOT/blend_frozen_gv21c.json" \
         "$OOF_ROOT/cov_oof_gv21.npy" "$OOF_ROOT/steps_gv21.npy" "$OOF_ROOT/dsc_gv21.json" "$YAUX_VAR" "$M80" "$PCA_DIR/pca_80.npz" "$PCA_DIR/pca_120.npz"; do
  [ -s "$f" ] || die "missing artefact $f"; done
if [ "$HEADS" = v22c10 ]; then for t in a b c d; do [ "$(ls "$HEAD22_ROOT"/ck_c10${t}_s*.pt 2>/dev/null | wc -l)" = 6 ] || die "missing v2.2 heads c10$t in $HEAD22_ROOT"; done
  [ -s "$LC/l_hp.py" ] || die "missing $LC/l_hp.py"; fi

# 1-2 features, sign / clock record, canonical step
# single-threaded BLAS in the feature workers (as the cache builders; N workers x N BLAS threads otherwise exhausts the thread limit)
FW=$(( NW > 140 ? 140 : NW ))
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$PK/pf_features.py" --in-dir "$IN_DIR" --out "$FEAT" --workers $FW | cut -c1-3000 || die features
[ -s "$FEAT/sign.npy" ] || die features; el features
python "$PK/pf_canon.py" --feat "$FEAT" --raw raw22 --scaler "$BASE_FEAT/scaler_v21.npz" --out "$FB" || die "canon base"
python "$PK/pf_canon.py" --feat "$FEAT" --raw raw21 --scaler "$HEAD_SCALER" --out "$FH" || die "canon heads"
cp "$YAUX_VAR" "$RUN/ccfeat/train/y_aux_var.json"
el canonical

# 3 FM
G=""; [ "$DEV" = cpu ] && G="CUDA_VISIBLE_DEVICES="
PSI='FULL_d30aux_s*+FULL_towerd30aux_s*+FULL_K120d30aux_s*'
env $G FEAT="$BASE_FEAT" OMP_NUM_THREADS=$TH TORCH_THREADS=$TH python "$PK/pf_fm.py" --feat-dir "$FB" --members-root "$BASE_ROOT/frame_mlp/members" \
  --psi "$PSI" --out "$RUN/fm.npz" --save-raw "$RUN/fm_raw" --device $DEV 2>&1 | tail -1
[ -s "$RUN/fm.npz" ] || die FM; el FM
# 4 TW
W=$BASE_ROOT/temporal/full/weights
( cd "$WS/stage11_run/ch1_temporal" && OMP_NUM_THREADS=$TH python predict_unseen.py --feat "$FB" --device $DEV --weights "$W/tR_w9d_s0.pt" "$W/tR_w9d_s1.pt" \
    "$W/tR_w9d_s2.pt" "$W/tR_w9d_s3.pt" --out "$RUN/tw.npz" --members-out "$RUN/tw_bag4.npz" 2>&1 | tail -1 )
[ -s "$RUN/tw.npz" ] || die TW; el TW
if [ "${DEVCHECK:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES= FEAT="$BASE_FEAT" OMP_NUM_THREADS=$TH TORCH_THREADS=$TH python "$PK/pf_fm.py" --feat-dir "$FB" --members-root "$BASE_ROOT/frame_mlp/members" \
    --psi "$PSI" --out "$RUN/fm_cpu.npz" --save-raw "$RUN/fm_raw_cpu" --device cpu 2>&1 | tail -1
  ( cd "$WS/stage11_run/ch1_temporal" && OMP_NUM_THREADS=$TH python predict_unseen.py --feat "$FB" --device cpu --weights "$W/tR_w9d_s0.pt" "$W/tR_w9d_s1.pt" \
      "$W/tR_w9d_s2.pt" "$W/tR_w9d_s3.pt" --out "$RUN/tw_cpu.npz" --members-out "$RUN/tw_bag4_cpu.npz" 2>&1 | tail -1 )
  el "devcheck (CPU FM / TW, comparison only)"
fi
# 5 SC
W=$BASE_ROOT/scalars/weights/full
( cd "$WS/stage11_run/ch1_scalars" && FEAT="$BASE_FEAT" OMP_NUM_THREADS=$TH python sc_predict.py --feat-dir "$FB" --out "$RUN/sc.npz" --weights \
  $W/gru_d10_fz_s0.pt $W/gru_d10_fz_s1.pt $W/la1_d10_e45_s0.pt $W/la1_d10_e45_s1.pt $W/la1_d10wd_s0.pt $W/la1_d10wd_s1.pt \
  $W/la1_d10_s0.pt $W/la1_d10_s1.pt $W/la1_d10_s2.pt $W/la1_d10_s3.pt $W/la1_d10_s4.pt $W/la1_d20_s0.pt $W/la1_d20_s1.pt \
  $W/la1_d10_phys_s0.pt $W/la1_d10_phys_s1.pt $W/ra1_d10_s0.pt $W/ra1_d10_s1.pt 2>&1 | tail -2 )
[ -s "$RUN/sc.npz" ] || die SC; el SC
# 6 blend
python "$PK/pf_blend.py" --fm "$RUN/fm.npz" --tw "$RUN/tw.npz" --sc "$RUN/sc.npz" --feat-dir "$FB" --out "$RUN/blend.npz" || die blend
el blend

# every CPU step below is a pool of single-threaded workers (run_correct.sh sets its own thread count for the GPU head inference)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 7 corrector (own per-run dir with links to the frozen artefacts)
HL=""; [ "$HEADS" = v21 ] && HL=$(ls "$HEAD_ROOT"/ck_direct2_s[0-5].pt)
for f in $HL "$HEAD_ROOT/blend_frozen_gv21c.json" "$OOF_ROOT/cov_oof_gv21.npy" "$OOF_ROOT/steps_gv21.npy" "$OOF_ROOT/dsc_gv21.json"; do
  ln -sfn "$(readlink -f "$f")" "$CCO/$(basename "$f")"; done
CCD=$WS/stage11_run/ch1_corrector; SPD=$WS/stage11_run/ch1d_specialists
if [ "$HEADS" = v22c10 ]; then      # heads on the base's v2.2 canonical rows (FB); run_correct.sh then finds hp_direct2_mlp_pkg.npy
  for t in a b c d; do
    sd=$(ls "$HEAD22_ROOT"/ck_c10${t}_s*.pt | sed 's/.*_s\([0-9]*\)\.pt/\1/' | sort -n | paste -sd,)
    env $G CC_OUT="$HEAD22_ROOT" OMP_NUM_THREADS=$TH python "$LC/l_hp.py" predict --tag c10$t --seeds "$sd" --model mlp --feat-dir "$FB" \
      --out "$CCO/hp_c10$t.npy" 2>&1 | tail -1
    [ -s "$CCO/hp_c10$t.npy" ] || die "heads c10$t"
  done
  python -c "import numpy as np; np.save('$CCO/hp_direct2_mlp_$SPLIT.npy', np.mean([np.load('$CCO/hp_c10%s.npy' % t) for t in 'abcd'], 0))" || die "head bag"
  el "heads v2.2 c10 (24 seeds)"
fi
echo "$HEADS" > "$RUN/heads_used.txt"     # read back by ch1/check_heads.sh
export CC_FEAT=$RUN/ccfeat CC_OUT=$CCO
(
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$CCO/tmp NW
  PRED=$RUN/blend.npz TAG=$TAG SPLIT=$SPLIT FAM=gv21 TGT=rob DIRECT=direct2 BLEND_FROZEN=blend_frozen_gv21c.json DSC=dsc_gv21.json \
    KEEP_BASE=volume CFG=r1e+09_e1_i13 bash "$CCD/run_correct.sh" 2>&1 | tail -6 | cut -c1-400
)
INCG=$CCO/preds_${SPLIT}_${TAG}_rob_r1e+09_e1_i13.npz
[ -s "$INCG" ] || die corrector; el corrector
# 8 robmi
echo "robmi metric $M80 sha256 $(sha256sum "$M80" | cut -c1-16)"
(
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$CCO/tmp
  cd "$SPD" && python sp_apply.py --split $SPLIT --tag $TAG --pred "$RUN/blend.npz" --name robmi --dhat "tgt_${SPLIT}_${TAG}_rob.npy" --direct --targets 7 \
    --prior metric --ridge trace --rs 1e9 --etas 1 --iters 13 --cov cov_oof_gv21.npy --dsc dsc_gv21.json --metric "$M80" \
    --workers $NW --write r1e+09_e1_i13 2>&1 | grep -E "metric prior|trace ridge|apply|wrote|Error|error" | cut -c1-400
)
R=$CCO/preds_${SPLIT}_${TAG}_robmi_r1e+09_e1_i13.npz
[ -s "$R" ] || die robmi; el robmi

# 9 final step (label-free)
LG=$WS/stage11_run/ch1d_liloss/li_guard.py
drv() { ( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=$CCO/tmp; cd "$CCD" && python cc_derive.py --pred "$1" --tag "$2" --split $SPLIT --workers $NW > "$CCO/logs/derive_$2.log" 2>&1 ) || die "derive $2"; }
nfail() { python -c "import numpy as np;a=np.load('$CCO/f11_${SPLIT}_$1.npy')[:,:7];print(int((~np.isfinite(a).all(1)).sum()))"; }
merge() { python "$SPD/sp_merge.py" --primary "$1" --fallback "$INCG" --f11-primary "$CCO/f11_${SPLIT}_$2.npy" --f11-fallback "$CCO/f11_${SPLIT}_fs_inc.npy" --feat-dir "$FB" --out "$3" | cut -c1-300 || die merge; }
FIN=$RUN/final.npz
if [ "$FINAL_STEP" = gM ]; then
  python "$LG" --pred "$R" --out "$RUN/robmi_g.npz" --workers $NW > "$RUN/guard1.log" 2>&1 || die "li guard"; head -1 "$RUN/guard1.log" | cut -c1-300
  drv "$RUN/robmi_g.npz" fs_g; N1=$(nfail fs_g)
  echo "final step gM: frames failing the scorer-path derivation after robmi -> li guard: $N1"
  if [ "$N1" = 0 ]; then
    ln -f "$RUN/robmi_g.npz" "$FIN" 2>/dev/null || cp "$RUN/robmi_g.npz" "$FIN"; cp "$RUN/robmi_g.npz.guard.json" "$FIN.guard.json"; cp "$CCO/f11_${SPLIT}_fs_g.npy" "$CCO/f11_${SPLIT}_fs_final.npy"
  else
    drv "$INCG" fs_inc; merge "$RUN/robmi_g.npz" fs_g "$RUN/robmi_gM0.npz"
    python "$LG" --pred "$RUN/robmi_gM0.npz" --out "$FIN" --workers $NW > "$RUN/guard2.log" 2>&1 || die "li guard 2"; head -1 "$RUN/guard2.log" | cut -c1-300
    drv "$FIN" fs_final
  fi
elif [ "$FINAL_STEP" = Mg ]; then
  drv "$R" fs_robmi; drv "$INCG" fs_inc; merge "$R" fs_robmi "$RUN/robmiF.npz"
  python "$LG" --pred "$RUN/robmiF.npz" --out "$FIN" --workers $NW > "$RUN/guard1.log" 2>&1 || die "li guard"; head -1 "$RUN/guard1.log" | cut -c1-300
  drv "$FIN" fs_final
else die "FINAL_STEP=$FINAL_STEP (gM | Mg)"; fi
NF=$(nfail fs_final)
python -c "import numpy as np;a=np.load('$CCO/f11_${SPLIT}_fs_final.npy')[:,:7];print('final census: frames',len(a),'non-finite per scalar',(~np.isfinite(a)).sum(0).tolist(),'li range [%.3f, %.3f]'%(np.nanmin(a[:,6]),np.nanmax(a[:,6])))"
[ "$NF" = 0 ] && echo "G6 zero derivation failures: PASS" || echo "G6_FAIL: $NF frames still fail the derivation (file is written; a person decides)"
cp "$FB/files.txt" "$FIN.files.txt"
python "$WS/stage11_run/ch1_blend/cb_verify.py" --corr "$FIN" --base "$RUN/blend.npz" --feat-dir "$FB" || echo VERIFY_FAILED
el "final step"

# 10 writer
python "$PK/write_submission.py" --pred "$FIN" --feat-dir "$FB" --order "$ORDER" --out "$RUN/submission.npz" || die writer
md5sum "$RUN/submission.npz"; el writer
echo "PREDICT_DONE $RUN/submission.npz wall_s $(( $(date +%s) - T0 ))"
# 11 gate (+ optional starter validator)
( cd "$WS/stage11_run/f_ch1_harden" && python final_gate.py --sub "$RUN/submission.npz" --in-dir "$IN_DIR" --order "$ORDER" --base "$RUN/blend.npz" --robmi "$R" --incgn "$INCG" \
    --guard-json "$FIN.guard.json" --clock v22 --report "$RUN/gate.json" --workers $NW > "$RUN/gate.log" 2>&1; echo "GATE_RC $?" )
grep -E "^\s*G[0-9]+|=== GATE" "$RUN/gate.log" | cut -c1-360
el gate
if [ -n "${VAL_CONFIG:-}" ]; then
  ( cd "$WS/starter" && python validate_submission.py "$RUN/submission.npz" --config "$VAL_CONFIG" --max-shots 0 ) > "$RUN/validate.log" 2>&1
  echo "VALIDATOR_RC $? | $(grep -v '^\s*$' "$RUN/validate.log" | tail -1 | cut -c1-300)"; el validator
fi
[ -n "${AFTER:-}" ] && bash -c "$AFTER"
echo "ALL_DONE wall_s $(( $(date +%s) - T0 )) g6_fail=$NF"
[ "$NF" = 0 ] || exit 3
