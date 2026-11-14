# Source this file. It defines the two roots every script uses:
#   WS   = <repository>/src   (code; also holds the two virtual environments and the pinned starter kit)
#   SCR  = work directory     (inputs, weights, generated caches, outputs; needs about 150 GB for training, 30 GB for inference)
_FEC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FEC_REPO=$_FEC_REPO
export WS=$FEC_REPO/src
export SCR=${SCR:-${FEC_WORK:-}}
[ -n "$SCR" ] || { echo "set FEC_WORK (or SCR) to a work directory" >&2; return 1 2>/dev/null || exit 1; }
mkdir -p "$SCR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
# one BLAS thread per process: every CPU stage is a pool of single-threaded workers (determinism + no oversubscription)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCR/.mplcache} XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCR/.xdg_cache} HF_HOME=${HF_HOME:-$SCR/.hf_home}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME"
