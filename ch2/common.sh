# Ch2 (MAST physics pipeline) shared settings. Sourced by the ch2 entrypoints after env.sh.
#
# The MAST pipeline is selected by a PROFILE: PIPE=<name> loads ch2/pipelines/<name>.sh (default: the pipeline named in
# ch2/pipelines/DEFAULT). A profile defines the solver, its configuration, the packer, optional post-processing stages and the
# DIII-D-derived constants (ch2/constants/<name>/). See ch2/pipelines/README.md for the profile interface.
export G1=$WS/stage11_run/ch2_G1 INV=$WS/stage11_run/ch2d_inverse INV6=$WS/stage11_run/e_mast_inv6 R7=$WS/stage11_run/g_restrict SCALC=$WS/stage11_run/ch2d_scalars
export GEOM=${GEOM:-$FEC_REPO/ch2/geometry}
export PY=$WS/.venv_gs/bin/python PYS=$WS/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NPROC=${NPROC:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
# Private temporary directory for this entrypoint and its children. Some schedulers give every job of a user the same node-local
# TMPDIR and remove it when any one of them ends; a process that creates a temporary file after that fails with FileNotFoundError
# (seen once in a rehearsal: the GPU training of the DIII-D twin). FEC_TMP_ROOT overrides the parent directory (default /tmp).
if [ -z "${FEC_TMP:-}" ] && FEC_TMP=$(mktemp -d "${FEC_TMP_ROOT:-/tmp}/fec.XXXXXXXX" 2>/dev/null); then
  export FEC_TMP TMPDIR=$FEC_TMP; trap 'rm -rf "$FEC_TMP"' EXIT
fi

PIPE=${PIPE:-$(tr -d ' \n' < "$FEC_REPO/ch2/pipelines/DEFAULT")}
export PIPE
PROFILE=$FEC_REPO/ch2/pipelines/$PIPE.sh
[ -f "$PROFILE" ] || { echo "unknown MAST pipeline '$PIPE' (profiles: $(cd "$FEC_REPO/ch2/pipelines" && ls *.sh | sed 's/\.sh$//' | tr '\n' ' '))"; exit 2; }
# profile interface (defaults; the profile overrides them)
TAG=""; SOLVER_DIR=""; SOLVER=""; SOLVE_ARGS=(); MAST_ARGS=(); PCAL_ARGS=(); PACKER=$INV6/pack_inv6.py; POST=(); MODELS=(); REF_MD5=""; CONST_PIPE=""
PSTAGES=(); RUN_MODELS=()   # profile stages (functions stage_<name>) and model files the run itself produces (paths relative to the run directory)
. "$PROFILE"
[ -n "$TAG" ] && [ -n "$SOLVER_DIR" ] && [ -n "$SOLVER" ] || { echo "profile $PROFILE must set TAG, SOLVER_DIR, SOLVER"; exit 2; }
export TAG
export CONST=${CONST:-$FEC_REPO/ch2/constants/${CONST_PIPE:-$PIPE}}   # CONST_PIPE: a profile that only adds post stages uses its base pipeline's constants

# run the profile's solver from its own directory (it imports its sibling modules); extra arguments are appended
run_solver() { ( cd "$SOLVER_DIR" && $PY "$SOLVER" "${SOLVE_ARGS[@]}" "$@" ); }

# a solved shard is complete when its summary exists, no shot ended in a fatal error and every listed shot has its part file
shard_ok() {  # <shard dir> <shot list>
  [ -s "$1/summary_${TAG}_mast.json" ] || return 1
  $PYS - "$1" "$2" "$TAG" <<'PY'
import json, re, sys
from pathlib import Path
s, l, t = sys.argv[1:4]
d = json.load(open(f"{s}/summary_{t}_mast.json"))
n = len([x for x in Path(l).read_text().split("\n") if x])
pat = re.compile(rf"^{re.escape(t)}_mast_\d{{5}}\.npz$")
parts = sum(1 for p in Path(s, "parts").iterdir() if pat.match(p.name)) if Path(s, "parts").is_dir() else 0
sys.exit(0 if d.get("n_fatal", 0) == 0 and d.get("n_shots") == n and parts == n else 1)
PY
}

# sha256 of every file that defines the selected pipeline: profile (+ the files it sources), constants, the json configs given to
# the solver, shipped models, and the model files the run itself produced (RUN_MODELS, e.g. a network trained on synthetic equilibria)
pipeline_record() {  # <out json> [<run dir>]
  local rd=${2:-.} rm=() m
  for m in "${RUN_MODELS[@]}"; do rm+=("$rd/$m"); done
  $PYS - "$1" "$PIPE" "$PROFILE" "$CONST" "${EXTRA_CONST:-}" "${#MODELS[@]}" "${MODELS[@]}" "${#rm[@]}" "${rm[@]}" "${SOLVE_ARGS[@]}" "${MAST_ARGS[@]}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
out, pipe, prof, const, xconst, nm, *rest = sys.argv[1:]
models, rest = rest[:int(nm)], rest[int(nm):]
nr = int(rest[0]); runm, args = rest[1:1 + nr], rest[1 + nr:]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
srcd, todo = [], [Path(prof)]                     # files the profile sources, recursively
while todo:
    for m in re.findall(r'ch2/pipelines/([A-Za-z0-9_.]+)"', todo.pop(0).read_text()):
        p = Path(prof).parent / m
        if p.is_file() and p not in srcd:
            srcd.append(p); todo.append(p)
rec = {"pipeline": pipe, "profile": Path(prof).name, "profile_sha256": h(prof),
       "profile_sources": {p.name: h(p) for p in srcd if p.is_file()},
       "constants": {p.name: h(p) for d in [const] + [x for x in xconst.split(":") if x] for p in sorted(Path(d).glob("*.json"))},
       "solver_configs": {Path(a).name: h(a) for a in args if a.endswith(".json") and Path(a).is_file()},
       "solver_sets": [a for a in args if "=" in a and not a.endswith(".json")],
       "models": {Path(m).name: h(m) for m in models},
       "run_models": {Path(m).name: (h(m) if Path(m).is_file() else None) for m in runm}}
json.dump(rec, open(out, "w"), indent=1)
PY
}

names_dir() {  # materialise a directory of empty <name>.parquet files from a list (only the sorted names are read by --exclude-holdout / --holdout-names)
  local list=$1 dir=$2; mkdir -p "$dir"; [ "$(ls "$dir" | wc -l)" = "$(grep -c . "$list")" ] || { rm -f "$dir"/*.parquet; while read -r n; do [ -n "$n" ] && : > "$dir/$n"; done < "$list"; }
}
