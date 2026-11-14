#!/bin/bash
# Section B of RUNBOOK_FINAL.md as SLURM jobs with dependencies: one call on a login node, from the frozen checkout. Every job runs
# exactly the command of its RUNBOOK step (the same entrypoints and arguments); only the scheduling is added.
#
#   FEC_WORK=<work dir> [settings] bash tools/final_slurm.sh submit    submit the steps (job ids -> $FEC_WORK/final_jobs/<RUN>/jobs.tsv)
#   FEC_WORK=<work dir> [settings] bash tools/final_slurm.sh check     the checks of RUNBOOK_FINAL.md section B on the run (PASS / FAIL lines)
#   FEC_WORK=<work dir> [settings] bash tools/final_slurm.sh status    per-job state, elapsed time, allocated cpu-minutes and node-hours
#
# Settings (environment; the defaults are the Final):
#   REPO=Sophelio/fusion-equilibrium-challenge-final  D3D=diii_d_private_test  MAST=mast_private_test  (expected shots N_D3D=879 N_MAST=1205)
#     dress rehearsal on the public test: REPO=Sophelio/fusion-equilibrium-challenge D3D=diii_d_public_test MAST=mast_public_test
#     (N_D3D=874 N_MAST=1206 are then taken automatically)
#   CH1_PROFILE, PIPE   DIII-D / MAST profiles (default: ch1/profiles/DEFAULT, ch2/pipelines/DEFAULT)
#   RUN=final           run name: DIII-D run $FEC_WORK/stage11_ch1/final_path/runs/<RUN>, MAST run $FEC_WORK/ch2_runs/<RUN>
#   DEVICE=gpu|cpu      gpu (default): the DIII-D predict and the network training get one GPU each (the devices of the Development files);
#                       cpu: no GPU job at all (CPU fallback, RUNBOOK_FINAL.md: valid files, not bit-identical to the GPU ones)
#   NSHARD=32           shards of the MAST solve and of the synthetic generation (one job each)
#   STEPS="..."         steps to submit (default: all): download order_d3d order_mast init solve syngen synreal syntrain ch1 finish
#                       val_d3d val_mast. A step left out is taken as done (its outputs must exist); dependencies on it are dropped.
#   HF_TOKEN_FILE=<f>   file holding a Hugging Face read token; read inside the jobs only, never printed (recommended: B.1, B.2, B.6)
#   REDO=1              with STEPS containing ch1 / finish whose earlier job already STARTED in this run (a run directory or output files
#                       exist): move those outputs aside (<dir>.redo_<UTC time>; nothing is deleted) before submitting. Without REDO=1
#                       `submit` refuses such a re-submission (the job itself would refuse to overwrite them, after waiting for its turn)
#   SLURM_PARTITION=workq; SB_EXTRA="<more sbatch options>"; DRYRUN=1 prints the sbatch commands only;
#   EXPECT_D3D_MD5 / EXPECT_MAST_MD5: md5 to compare with in `check` (a rehearsal on the public test: the Development files)
# Before any job is submitted, `submit` checks (PREFLIGHT lines) the checkout's environments and starter kit, `src/.vendor_scorer` for the
# int_* / dec_* profiles, and every weight file of the selected profiles against its manifest (sizes + sha256 under $FEC_WORK: the
# DIII-D profile's manifests, the MAST profile's model files in weights/MANIFEST_CH2.sha256); it submits nothing if one is missing or differs.
# Job shapes (CPUS_<step>, GPUS_<step>, MEM_<step>, TIME_<step> override, e.g. TIME_CH1=01:30:00): requests stay at or below 100 GB per
# 72 cores (exception: the CPU-path DIII-D job, 160 GB as measured) - the allocation is billed by the largest of the core, GPU and memory
# shares of a node (288 cores, 4 GPUs, about 450 GB per node); the scheduler books cores x time limit while a job runs.
set -uo pipefail
CMD=${1:-help}
if [ "$CMD" = __job ]; then                       # inside a job: the settings of the submission travel in a file (sbatch runs a copy of this script)
  SETF=${2:?settings}; STEP=${3:?step}; K=${4:-}
  unset WS SCR PYTHONPATH VIRTUAL_ENV
  . "$SETF"
else
  FEC_REPO=$(cd "$(dirname "$0")/.." && pwd)
  unset SCR          # the work directory is FEC_WORK, as inside the jobs (env.sh would otherwise prefer an SCR of the login shell)
fi
. "$FEC_REPO/env.sh" || exit 1
REPO=${REPO:-Sophelio/fusion-equilibrium-challenge-final}; D3D=${D3D:-diii_d_private_test}; MAST=${MAST:-mast_private_test}
case $D3D in diii_d_private_test) N_D3D=${N_D3D:-879};; diii_d_public_test) N_D3D=${N_D3D:-874};; *) N_D3D=${N_D3D:-0};; esac
case $MAST in mast_private_test) N_MAST=${N_MAST:-1205};; mast_public_test) N_MAST=${N_MAST:-1206};; *) N_MAST=${N_MAST:-0};; esac
RUN=${RUN:-final}; DEVICE=${DEVICE:-gpu}; NSHARD=${NSHARD:-32}
CH1_PROFILE=${CH1_PROFILE:-$(tr -d ' \n' < "$FEC_REPO/ch1/profiles/DEFAULT")}; PIPE=${PIPE:-$(tr -d ' \n' < "$FEC_REPO/ch2/pipelines/DEFAULT")}
PY=$WS/.venv/bin/python
PYC=$PY; [ -x "$PYC" ] || PYC=python3      # check / status only (standard library)
IN_D=$FEC_WORK/in/d3d; IN_M=$FEC_WORK/in/mast; ORD_D=$FEC_WORK/order/d3d; ORD_M=$FEC_WORK/order/mast
C1RUN=$SCR/stage11_ch1/final_path/runs/$RUN; C2RUN=$SCR/ch2_runs/$RUN
JD=$FEC_WORK/final_jobs/$RUN; LOGD=$JD/logs
ALL_STEPS="download order_d3d order_mast init solve syngen synreal syntrain ch1 finish val_d3d val_mast"
case $DEVICE in gpu|cpu) ;; *) echo "DEVICE=gpu|cpu"; exit 2;; esac

# ---------------------------------------------------------------------------------------------------------------------------------
run_step() {   # the RUNBOOK command of one step (cwd = checkout)
  local s=$1 k=$2
  case $s in
    download)   # B.1
      $PY tools/download_inputs.py --repo "$REPO" --config "$D3D" --out "$IN_D" || return 1
      $PY tools/download_inputs.py --repo "$REPO" --config "$MAST" --out "$IN_M" || return 1
      local nd nm; nd=$(ls "$IN_D" | grep -c '\.parquet$'); nm=$(ls "$IN_M" | grep -c '\.parquet$')
      echo "COUNT $D3D $nd expected $N_D3D; $MAST $nm expected $N_MAST"
      [ "$nd" = "$N_D3D" ] && [ "$nm" = "$N_MAST" ] || { echo "COUNT_MISMATCH: incomplete download"; return 1; };;
    order_d3d)  # B.2
      $PY tools/stream_order.py --shot-dir "$IN_D" --repo "$REPO" --config "$D3D" --out-dir "$ORD_D";;
    order_mast)
      $PY tools/stream_order.py --shot-dir "$IN_M" --repo "$REPO" --config "$MAST" --out-dir "$ORD_M";;
    init)       # run directory, shot list and shard lists written once, before the parallel shard jobs
      PIPE=$PIPE NSHARD=$NSHARD STAGES=init bash ch2/predict_mast.sh "$IN_M" "$RUN" sorted "$MAST.npz" || return 1
      [ "$(ls "$C2RUN/shards" | grep -c '^list_')" = "$NSHARD" ];;
    solve|syngen)   # B.3, one shard
      PIPE=$PIPE NSHARD=$NSHARD SHARD=$k STAGES=$s bash ch2/predict_mast.sh "$IN_M" "$RUN" sorted "$MAST.npz";;
    synreal)
      PIPE=$PIPE NSHARD=$NSHARD STAGES=synreal bash ch2/predict_mast.sh "$IN_M" "$RUN" sorted "$MAST.npz";;
    syntrain)   # B.4
      if [ "$DEVICE" = gpu ]; then
        [ "$($PY -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null)" = 1 ] || { echo "NO_GPU_VISIBLE in a GPU job: stop"; return 1; }
      else export CUDA_VISIBLE_DEVICES=; fi
      PIPE=$PIPE NSHARD=$NSHARD STAGES=syntrain bash ch2/predict_mast.sh "$IN_M" "$RUN" sorted "$MAST.npz";;
    ch1)        # B.5 DIII-D
      if [ "$DEVICE" = gpu ]; then export DEV=cuda; else export DEV=cpu CUDA_VISIBLE_DEVICES=; fi
      CH1_PROFILE=$CH1_PROFILE bash ch1/predict.sh "$IN_D" "$RUN" "$ORD_D/stream_order.txt" "$D3D.npz";;
    finish)     # B.5 MAST finish
      PIPE=$PIPE NSHARD=$NSHARD STAGES="merge scal pack post validate" bash ch2/predict_mast.sh "$IN_M" "$RUN" "$ORD_M/stream_order.txt" "$MAST.npz";;
    val_d3d)    # B.6
      $PY tools/validate_final.py "$C1RUN/$D3D.npz" --repo "$REPO" --config "$D3D";;
    val_mast)
      $PY tools/validate_final.py "$C2RUN/submission/$MAST.npz" --repo "$REPO" --config "$MAST";;
    *) echo "unknown step $s"; return 2;;
  esac
}

if [ "$CMD" = __job ]; then
  cd "$FEC_REPO" || exit 1
  if [ -n "${HF_TOKEN_FILE:-}" ] && [ -s "$HF_TOKEN_FILE" ]; then HF_TOKEN=$(tr -d ' \n' < "$HF_TOKEN_FILE"); export HF_TOKEN; fi
  echo "=== final_slurm $STEP${K:+ $K} job ${SLURM_JOB_ID:-local} $(hostname) $(date -u +%FT%TZ) cpus=${SLURM_CPUS_PER_TASK:-?} gpus=${SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-0}} run=$RUN device=$DEVICE repo=$REPO d3d=$D3D mast=$MAST ch1=$CH1_PROFILE pipe=$PIPE nshard=$NSHARD hf_token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo none)"
  T0=$(date +%s)
  run_step "$STEP" "$K"; rc=$?
  echo "=== STEP_END $STEP${K:+ $K} rc=$rc wall_s $(( $(date +%s) - T0 )) $(date -u +%FT%TZ)"
  exit $rc
fi

# ---------------------------------------------------------------------------------------------------------------------------------
shape() {   # <step> -> "cpus gpus mem time"; time limits = about 1.5 x the longest measured run of the step (rehearsals of 12-15 October)
  local s=$1 c=72 g=0 m=100G t=00:20:00
  case $s in
    download)   t=00:45:00;;                  # 356 s in the dress rehearsal (10.4 GB); Final-day Hub load unknown
    order_d3d)  t=01:15:00;;                  # 39-41 min measured; one run hit a 60-min limit (Hub throughput)
    order_mast) t=00:55:00;;                  # 21-36 min measured (unauthenticated streaming)
    init)       t=00:05:00;;                  # seconds
    solve)      t=00:22:00;;                  # p7li shards: mean 591 s, max 883 s
    syngen)     t=00:18:00;;                  # mean 533 s, max 717 s
    synreal)    t=00:05:00;;                  # 18 s
    syntrain)   if [ "$DEVICE" = gpu ]; then g=1; t=00:10:00; else t=00:30:00; fi;;    # GPU 245-251 s; CPU 902 s training + prediction
    ch1)        if [ "$DEVICE" = gpu ]; then c=288; g=1; m=400G; t=01:10:00             # 2,281-2,679 s on 288 cores + 1 GPU
                else m=160G; t=02:50:00; fi;;                                           # CPU: 6,677 s on 72 cores; 160 GB is the measured
                                                                                        # working request (the only one above 100 GB per 72 cores)
    finish)     t=00:20:00;;                  # merge + scal + pack + post (synblend, dec, corr) + local check: 539 s
    val_d3d)    t=00:55:00;;                  # 23-34 min measured (one Hub read time-out retried)
    val_mast)   t=00:55:00;;                  # 18-36 min measured
  esac
  local u; u=$(echo "$s" | tr "[:lower:]" "[:upper:]"); eval "c=\${CPUS_$u:-$c}; g=\${GPUS_$u:-$g}; m=\${MEM_$u:-$m}; t=\${TIME_$u:-$t}"
  echo "$c $g $m $t"
}

if [ "$CMD" = submit ]; then
  STEPS=${STEPS:-$ALL_STEPS}
  PST=$( set +u; . "$FEC_REPO/ch2/common.sh" >/dev/null 2>&1; echo "${PSTAGES[*]}" )   # profile stages of the MAST profile
  [ -n "$PST" ] || echo "MAST profile $PIPE has no profile stages (inverse profile): no syngen / synreal / syntrain jobs"
  [[ " $PST " == *" syngen "* ]] || STEPS=$(echo " $STEPS " | sed 's/ syngen / /; s/ synreal / /; s/ syntrain / /')
  mkdir -p "$LOGD"
  SETF=$JD/settings_$(date -u +%Y%m%dT%H%M%S).env
  { for v in FEC_REPO FEC_WORK REPO D3D MAST N_D3D N_MAST RUN DEVICE NSHARD CH1_PROFILE PIPE HF_TOKEN_FILE; do printf 'export %s=%q\n' "$v" "${!v:-}"; done; } > "$SETF"
  echo "settings $SETF:"; sed 's/^export /  /' "$SETF"
  has() { [[ " $STEPS " == *" $1 "* ]]; }
  # --- preflight: environments, starter kit, scorer overlay and every weight file of the selected profiles (nothing is submitted otherwise)
  PF=0; pf() { echo "PREFLIGHT $*"; }
  for f in "$WS/.venv/bin/python" "$WS/.venv_gs/bin/python" "$WS/starter/validate_submission.py"; do
    [ -e "$f" ] || { pf "FAIL missing $f (env/build_env.sh)"; PF=1; }; done
  [ -f "$FEC_REPO/ch1/profiles/$CH1_PROFILE.sh" ] || { pf "FAIL unknown DIII-D profile $CH1_PROFILE"; PF=1; }
  [ -f "$FEC_REPO/ch2/pipelines/$PIPE.sh" ] || { pf "FAIL unknown MAST profile $PIPE"; PF=1; }
  if has ch1; then
    CHM=$( set +u; . "$FEC_REPO/ch1/common.sh" >/dev/null 2>&1; echo "$CH1_MANIFESTS" )
    [ -n "$CHM" ] || { pf "FAIL DIII-D profile $CH1_PROFILE: no manifest list"; PF=1; }
    for m in $CHM; do
      o=$("$PYC" "$FEC_REPO/tools/verify_manifest.py" "$FEC_REPO/$m" "$SCR" 2>&1); r=$?
      [ $r = 0 ] && pf "OK DIII-D $CH1_PROFILE: $(echo "$o" | tail -1)" || { pf "FAIL DIII-D $CH1_PROFILE: $(echo "$o" | grep -E 'MISSING|MISMATCH|manifest' | head -5 | tr '\n' ';')"; PF=1; }
    done
  fi
  if has init || has solve || has syngen || has synreal || has syntrain || has finish; then
    MOD=$( set +u; . "$FEC_REPO/ch2/common.sh" >/dev/null 2>&1; printf '%s\n' "${MODELS[@]}"; [[ "$PIPE" == int_* || "$PIPE" == dec_* ]] && echo "@VENDOR" )
    TM=$(mktemp); nmod=0
    while read -r f; do
      [ -n "$f" ] || continue
      if [ "$f" = @VENDOR ]; then [ -d "$WS/.vendor_scorer/skimage" ] || { pf "FAIL MAST $PIPE needs $WS/.vendor_scorer (env/build_env.sh)"; PF=1; }; continue; fi
      rel=${f#"$SCR"/}; nmod=$((nmod + 1))
      grep -E "^[0-9]+ [0-9a-f]{64} $rel\$" "$FEC_REPO/weights/MANIFEST_CH2.sha256" >> "$TM" || { pf "FAIL MAST $PIPE: model file $f is not listed in weights/MANIFEST_CH2.sha256"; PF=1; }
    done <<< "$MOD"
    if [ -s "$TM" ]; then
      o=$("$PYC" "$FEC_REPO/tools/verify_manifest.py" "$TM" "$SCR" 2>&1); r=$?
      [ $r = 0 ] && pf "OK MAST $PIPE: $nmod model file(s): $(echo "$o" | tail -1 | sed 's/^manifest [^:]*: //')" || { pf "FAIL MAST $PIPE: $(echo "$o" | grep -E 'MISSING|MISMATCH' | head -5 | tr '\n' ';')"; PF=1; }
    elif [ $nmod = 0 ]; then pf "OK MAST $PIPE: no model file"; fi
    rm -f "$TM"
  fi
  if [ $PF != 0 ]; then
    if [ "${DRYRUN:-0}" = 1 ]; then echo "PREFLIGHT_FAILED (DRYRUN: the commands are printed anyway; a real submit stops here)"
    else echo "PREFLIGHT_FAILED: nothing submitted (tools/fetch_weights.sh into this FEC_WORK, env/build_env.sh)"; exit 3; fi
  else echo "PREFLIGHT OK"; fi
  # --- a step that already started in this run refuses to overwrite its outputs: move them aside (REDO=1) or stop here
  TS=$(date -u +%Y%m%dT%H%M%S); RB=()
  has ch1 && [ -e "$C1RUN" ] && RB+=("$C1RUN")
  if has finish; then for f in "$C2RUN/submission/$MAST.npz" "$C2RUN"/post/*"_$MAST.npz"; do [ -e "$f" ] && { RB+=("$C2RUN/post" "$C2RUN/submission"); break; }; done; fi
  if [ ${#RB[@]} -gt 0 ]; then
    if [ "${REDO:-0}" = 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
      for d in "${RB[@]}"; do mv "$d" "$d.redo_$TS" && echo "REDO moved aside: $d -> $d.redo_$TS" || { echo "REDO: cannot move $d"; exit 3; }; done
    elif [ "${DRYRUN:-0}" = 1 ]; then echo "STARTED_BEFORE: ${RB[*]} (a real submit needs REDO=1)"
    else echo "STARTED_BEFORE: ${RB[*]} exist(s) from an earlier job of run $RUN; re-run with REDO=1 to move them aside (nothing is deleted)"; exit 3; fi
  fi
  dep() {  # tokens: <step> = afterok on that step (if submitted in this call); -- = the following steps afterany; @<id> = afterok on a job id
    local mode=ok okl="" anyl="" s j out=""
    for s in "$@"; do
      case $s in --) mode=any;; @*) okl="$okl:${s#@}";; *) eval "j=\${J_$s:-}"; [ -n "$j" ] && { [ $mode = ok ] && okl="$okl:$j" || anyl="$anyl:$j"; };; esac
    done
    [ -n "$okl" ] && out="afterok$okl"; [ -n "$anyl" ] && out="${out:+$out,}afterany$anyl"
    [ -n "$out" ] && echo "--dependency=$out"
    return 0
  }
  LAST=""
  sub() {  # <step> <k|-> [dependency option] -> job id in LAST
    local s=$1 k=$2 d=${3:-} c g m t name kk=""
    [ "$k" = - ] || kk=$k
    read -r c g m t <<< "$(shape "$s")"
    name=F_${RUN}_$s$kk
    local args=(-J "$name" -p "${SLURM_PARTITION:-workq}" -N 1 --cpus-per-task="$c" --mem="$m" -t "$t" -o "$LOGD/%x_%j.out" --export=NONE)
    [ "$g" != 0 ] && args+=(--gpus="$g")
    [ -n "$d" ] && args+=("$d")
    # shellcheck disable=SC2206
    [ -n "${SB_EXTRA:-}" ] && args+=($SB_EXTRA)
    if [ "${DRYRUN:-0}" = 1 ]; then echo "sbatch ${args[*]} tools/final_slurm.sh __job $SETF $s $kk"; LAST=DRY_$name
    else LAST=$(sbatch --parsable "${args[@]}" "$FEC_REPO/tools/final_slurm.sh" __job "$SETF" "$s" $kk) && [ -n "$LAST" ] || { echo "SUBMIT_FAILED $name (jobs already submitted: $JD/jobs.tsv)"; exit 1; }; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LAST" "$name" "$s" "$kk" "$c" "$g" "$m" "$t" >> "$JD/jobs.tsv"
    echo "  $s${kk:+ $kk}: $LAST  ($c cores, $g GPU, $m, -t $t${d:+, $d})"
  }
  one() { sub "$@"; eval "J_$1=\$LAST"; }
  has download   && one download - "$(dep download)"
  has order_d3d  && one order_d3d - "$(dep download)"
  has order_mast && one order_mast - "$(dep download -- order_d3d)"          # one Hub stream at a time
  has init       && one init - "$(dep download)"
  SH=(); GN=()
  if has solve;  then for k in $(seq 0 $((NSHARD - 1))); do sub solve "$k" "$(dep download init)"; SH+=("@$LAST"); done; fi
  if has syngen; then for k in $(seq 0 $((NSHARD - 1))); do sub syngen "$k" "$(dep download init)"; GN+=("@$LAST"); done; fi
  has synreal    && one synreal - "$(dep download init)"
  has syntrain   && one syntrain - "$(dep synreal ${GN[@]+"${GN[@]}"})"
  has ch1        && one ch1 - "$(dep download order_d3d)"
  has finish     && one finish - "$(dep order_mast init syntrain ${SH[@]+"${SH[@]}"})"
  has val_d3d    && one val_d3d - "$(dep ch1)"
  has val_mast   && one val_mast - "$(dep finish -- val_d3d)"                 # the validators run one after the other
  echo "SUBMITTED $(grep -c . "$JD/jobs.tsv") jobs in total for run $RUN -> $JD/jobs.tsv; logs $LOGD"
  exit 0
fi

if [ "$CMD" = status ]; then
  [ -s "$JD/jobs.tsv" ] || { echo "no jobs.tsv in $JD"; exit 1; }
  ids=$(cut -f1 "$JD/jobs.tsv" | grep -v '^DRY_' | paste -sd,)
  NODE_MB=${NODE_MEM_MB:-$(sinfo -h -p "${SLURM_PARTITION:-workq}" -o %m 2>/dev/null | sort -n | tail -1)}
  SA=$(mktemp); sacct -X -n -P -j "$ids" -o JobID,JobName,State,ElapsedRaw,AllocCPUS,AllocTRES,CPUTimeRAW,Start,End > "$SA"
  $PYC - "$NODE_MB" "$JD/jobs.tsv" "$SA" <<'PY'
import sys, re, collections
node_mb = float(sys.argv[1] or 0)
stepof = {l.split("\t")[0]: l.split("\t")[2] for l in open(sys.argv[2]) if l.strip()}
rows = [l for l in open(sys.argv[3]) if l.strip()]
tot = collections.defaultdict(float); grp = collections.defaultdict(lambda: [None, None, 0.0, 0.0, 0, collections.Counter()])
print(f"{'job':>9} {'name':28s} {'state':10s} {'elapsed_s':>9} {'cpus':>5} {'gpus':>4} {'mem_GB':>7} {'cpu_min':>8} {'NHR':>7}")
for ln in rows:
    jid, name, st, el, cpus, tres, ct, start, end = ln.rstrip("\n").split("|")
    el, cpus, ct = int(el or 0), int(cpus or 0), int(ct or 0)
    g = int(re.search(r"gres/gpu=(\d+)", tres).group(1)) if "gres/gpu=" in tres else 0
    mm = re.search(r"mem=([\d.]+)([MGT])", tres); mem = float(mm.group(1)) * {"M": 1 / 1024, "G": 1, "T": 1024}[mm.group(2)] if mm else 0.0
    share = max(cpus / 288, g / 4, (mem * 1024 / node_mb) if node_mb else 0)
    nhr = share * el / 3600
    print(f"{jid:>9} {name:28s} {st[:10]:10s} {el:9d} {cpus:5d} {g:4d} {mem:7.0f} {ct / 60:8.0f} {nhr:7.3f}")
    step = stepof.get(jid, name)
    r = grp[step]
    if start not in ("", "Unknown", "None"): r[0] = min(r[0] or start, start)
    if end not in ("", "Unknown", "None"): r[1] = max(r[1] or end, end)
    r[2] += ct / 60; r[3] += nhr; r[4] += 1; r[5][st.split()[0]] += 1
    tot["cpu_min"] += ct / 60; tot["nhr"] += nhr
print(f"\n{'step':10s} {'jobs':>4} {'first start':20s} {'last end':20s} {'cpu_min':>8} {'NHR':>7}  states")
for s, r in grp.items():
    print(f"{s:10s} {r[4]:4d} {str(r[0]):20s} {str(r[1]):20s} {r[2]:8.0f} {r[3]:7.3f}  {dict(r[5])}")
print(f"TOTAL cpu_min {tot['cpu_min']:.0f}  NHR {tot['nhr']:.3f}  (NHR = max(cores/288, GPUs/4, memory/node memory) x elapsed hours; node memory {node_mb:.0f} MB)")
PY
  rc=$?; rm -f "$SA"; exit $rc
fi

if [ "$CMD" = check ]; then
  cd "$FEC_REPO" || exit 1
  REFREC=$(mktemp)      # record of the MAST profile recomputed from THIS checkout (and the run's own model files) for the record row
  ( set +u; . "$FEC_REPO/ch2/common.sh" >/dev/null 2>&1 && pipeline_record "$REFREC" "$C2RUN" ) >/dev/null 2>&1
  $PYC - "$LOGD" "$RUN" "$CH1_PROFILE" "$PIPE" "$N_D3D" "$N_MAST" "$NSHARD" "$C1RUN" "$C2RUN" "$D3D" "$MAST" "$IN_D" "$IN_M" "$ORD_D" "$ORD_M" \
      "${EXPECT_D3D_MD5:-}" "${EXPECT_MAST_MD5:-}" "$PY" "$WS" "$REFREC" "$FEC_REPO/weights/MANIFEST_CH2.sha256" <<'PY'
import glob, hashlib, json, os, re, subprocess, sys
(logd, run, c1p, pipe, nd, nm, nsh, c1run, c2run, d3d, mast, ind, inm, ordd, ordm, exp_d, exp_m, py, ws, refrec, man2) = sys.argv[1:]
nd, nm, nsh = int(nd), int(nm), int(nsh)
res = []
def ok(name, cond, info=""):
    res.append(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + info) if info else ''}")
def last(step):
    fs = sorted(glob.glob(f"{logd}/F_{run}_{step}_*.out"), key=os.path.getmtime)
    return open(fs[-1], errors="replace").read() if fs else ""
def lastk(step, k):
    fs = sorted(glob.glob(f"{logd}/F_{run}_{step}{k}_*.out"), key=os.path.getmtime)
    return open(fs[-1], errors="replace").read() if fs else ""
def md5(p):
    h = hashlib.md5(); f = open(p, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
# B.1 / B.2
n_d = len(glob.glob(f"{ind}/*.parquet")); n_m = len(glob.glob(f"{inm}/*.parquet"))
ok("shot counts (parquet files)", n_d == nd and n_m == nm, f"{d3d} {n_d} / {nd}, {mast} {n_m} / {nm}")
L = last("download"); ok("download log", L.count("DOWNLOAD_OK") == 2 and "rc=0" in L, "DOWNLOAD_OK x %d" % L.count("DOWNLOAD_OK"))
for nm_, od in (("order_d3d", ordd), ("order_mast", ordm)):
    L = last(nm_); m = re.search(r"ORDER_CHECK (\w+): (\{.*\})", L)
    info = json.loads(m.group(2)) if m else {}
    ok(f"stream order {nm_}", bool(m) and m.group(1) == "OK" and os.path.isfile(f"{od}/stream_order.txt"),
       f"n_stream {info.get('n_stream')} identical_to_sorted {info.get('identical_to_sorted')} {info.get('seconds')} s")
# B.5 DIII-D
L = last("ch1")
pos = lambda s: L.find(s)
seq = [f"=== ch1 predict profile={c1p} ", "HEADS_CHECK OK: profile " + c1p, "ALL_DONE"]
if c1p == "hc10x24": seq.insert(1, "heads v2.2 c10 (24 seeds)")
p = [pos(s) for s in seq]
ok("DIII-D log lines in order", all(x >= 0 for x in p) and p == sorted(p), " -> ".join(seq))
ok("DIII-D run complete", "PREDICT_INCOMPLETE" not in L and "PREDICT_FAILED" not in L and "STEP_END ch1 rc=0" in L)
dev = re.search(r"DEVICE: [^\n]*", L); print(f"[INFO] {dev.group(0) if dev else 'DEVICE line missing'}")
prof = open(f"{c1run}/profile.txt").read().strip() if os.path.isfile(f"{c1run}/profile.txt") else None
ok("DIII-D profile.txt", prof == c1p, str(prof))
G = open(f"{c1run}/gate.log", errors="replace").read() if os.path.isfile(f"{c1run}/gate.log") else ""
mg = re.search(r"=== GATE (\w+): (\d+) PASS, (\d+) REVIEW, (\d+) FAIL", G)
ok("DIII-D gate 0 FAIL", bool(mg) and mg.group(4) == "0", mg.group(0) if mg else "no gate summary")
ok("DIII-D gate G5 order PASS", bool(re.search(r"\[PASS\s*\] G5 order", G)))
for ln in re.findall(r"\[(?:REVIEW|FAIL)\s*\] [^\n]*", G): print(f"[INFO] gate {ln[:200]}")
ok("DIII-D G6 zero derivation failures", "G6 zero derivation failures: PASS" in L)
fd = f"{c1run}/{d3d}.npz"
if os.path.isfile(fd):
    m5 = md5(fd); print(f"[INFO] {d3d}.npz md5 {m5} bytes {os.path.getsize(fd)}")
    if exp_d: ok("DIII-D md5 = expected", m5 == exp_d, f"expected {exp_d}")
else: ok("DIII-D file", False, fd)
# B.3 / B.4 / B.5 MAST
bad = [k for k in range(nsh) if "STEP_END solve %d rc=0" % k not in lastk("solve", k)]
ok("MAST solve shards", not bad, f"{nsh - len(bad)} / {nsh} rc=0" + (f"; not ok {bad}" if bad else ""))
if glob.glob(f"{logd}/F_{run}_syngen*_*.out"):
    bad = [k for k in range(nsh) if "STEP_END syngen %d rc=0" % k not in lastk("syngen", k)]
    ok("MAST syngen shards", not bad, f"{nsh - len(bad)} / {nsh} rc=0" + (f"; not ok {bad}" if bad else ""))
    L = last("syntrain"); mm = re.search(r"model ([0-9a-f]{64})", L); mp = re.search(r'PREDICT \{[^\n]*?"n_shots": (\d+)', L)
    nshots = int(mp.group(1)) if mp else None          # the stage prints the PREDICT line cut to 300 characters
    dl = re.search(r"syntrain: device [^\n]*", L); dl = dl.group(0) if dl else "no device line"
    ok("network trained + every shot predicted", bool(mm) and nshots == nm and "STEP_END syntrain rc=0" in L,
       f"model sha256 {mm.group(1)[:16] if mm else None}…, n_shots {nshots}; {dl}")
F = last("finish")
mo = re.search(r"MERGE_OK (\d+) shots", F); ok("MAST merge", bool(mo) and int(mo.group(1)) == nm, mo.group(0) if mo else "no MERGE_OK")
if "dec" in pipe or "int_" in pipe:
    kt = re.findall(r"KTAB mast (\S+): (\d+) frames, (\d+) shots[^\n]*w contour fail ([\d.]+)", F)
    ok("corrector / decoder tables cover every shot", bool(kt) and all(int(s) == nm for _, _, s, _ in kt), "; ".join(f"{t} {f} fr {s} sh wfail {w}" for t, f, s, w in kt))
    ka = re.search(r"KAPPLY (\{.*)", F); ok("corrector applied to every frame", bool(ka) and "CORR_SHOTS selects" not in F, (ka.group(1)[:160] if ka else "no KAPPLY"))
if "dec" in pipe:
    dr = re.search(r"DECRUN (\{[^\n]*)", F); st = {}
    if dr:
        print(f"[INFO] DECRUN {dr.group(1)[:400]}")
        m = re.search(r'"n_frames": (\d+), "status": (\{[^}]*\})', dr.group(1)); cm = re.search(r'"closure_max": ([0-9.eE+-]+)', dr.group(1))
        if m:
            st = json.loads(m.group(2)); nf = int(m.group(1))
    ok("decoder run: every frame accounted for, closure < 1e-10", bool(st) and sum(st.values()) == nf and bool(cm) and float(cm.group(1)) < 1e-10,
       f"status {st}, closure_max {cm.group(1) if dr and cm else None}")
    for t in ("DWEST", "DSUP", "DWAPPLY"):
        m = re.search(t + r" [^\n]*", F); print(f"[INFO] {m.group(0)[:300] if m else t + ' missing'}")
for ln in re.findall(r"post \S+ -> [0-9a-f]{32}", F): print(f"[INFO] {ln}")
vl = re.search(r"validate_local rc=(\d+)[^\n]*", F); print(f"[INFO] finish job: {vl.group(0)[:200] if vl else 'no validate_local line'}")
fm = f"{c2run}/submission/{mast}.npz"
if os.path.isfile(fm) and os.path.isfile(f"{ordm}/stream_order.txt"):   # shot i of the file against the input at stream index i
    v = subprocess.run([py if os.path.exists(py) else sys.executable, f"{ws}/stage6_run/variants/P_ops_mast_package/validate_local.py", fm, "--shot-dir", inm, "--machine", "MAST",
                        "--order", f"{ordm}/stream_order.txt"], capture_output=True, text=True)
    vt = [x for x in (v.stdout + v.stderr).strip().split("\n") if x.strip()] or [""]
    ok("MAST local format check (stream order)", v.returncode == 0, " | ".join([vt[0][:150], vt[-1][:150]]))
else: ok("MAST local format check (stream order)", False, "file or stream order missing")
ok("MAST finish complete", "CH2_PREDICT_DONE" in F and "STEP_END finish rc=0" in F)
if os.path.isfile(fm):
    m5 = md5(fm); print(f"[INFO] {mast}.npz md5 {m5} bytes {os.path.getsize(fm)}")
    if exp_m: ok("MAST md5 = expected", m5 == exp_m, f"expected {exp_m}")
    rj = f"{c2run}/submission/record.json"
    if os.path.isfile(rj):
        # every hash of the run's record = the same record recomputed from this checkout; shipped models = weights/MANIFEST_CH2.sha256
        r = json.load(open(rj)); ref = json.load(open(refrec)) if os.path.getsize(refrec) else {}
        diff = [k for k in sorted(set(r) | set(ref)) if r.get(k) != ref.get(k)]
        man = {}
        for ln in open(man2):
            f_ = ln.split()
            if len(f_) == 3 and not ln.startswith("#"): man[os.path.basename(f_[2])] = f_[1]
        mbad = [n for n, h in r.get("models", {}).items() if man.get(n) != h]
        nh = 1 + sum(len(r.get(k) or {}) for k in ("profile_sources", "constants", "solver_configs", "models", "run_models"))
        ok("record.json = frozen package", bool(ref) and r.get("pipeline") == pipe and not diff and not mbad,
           f"{r.get('pipeline')}: {nh} sha256 compared (profile, sources {len(r.get('profile_sources', {}))}, constants {len(r.get('constants', {}))}, "
           f"solver configs {len(r.get('solver_configs', {}))}, models {len(r.get('models', {}))} = manifest, run models {len(r.get('run_models') or {})})"
           + (f"; DIFFERENT: {diff}" if diff else "") + (f"; models not = manifest: {mbad}" if mbad else "") + ("" if ref else "; no reference record"))
        for n, h in (r.get("run_models") or {}).items(): print(f"[INFO] run model {n} sha256 {h}")
    else: ok("record.json = frozen package", False, "no record.json")
else: ok("MAST file", False, fm)
# B.6
for v, nm_ in (("val_d3d", d3d), ("val_mast", mast)):
    L = last(v); okv = "✓ OK" in L and f"STEP_END {v} rc=0" in L
    ok(f"official validator {nm_}", okv, f"{L.count('429')} lines with 429; " + (re.findall(r"[^\n]*✓[^\n]*", L) or ["no OK line"])[-1][:120])
print(f"CHECK_SUMMARY {sum(res)} PASS, {len(res) - sum(res)} FAIL")
sys.exit(0 if all(res) else 1)
PY
  rc=$?; rm -f "$REFREC"; exit $rc
fi

sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
