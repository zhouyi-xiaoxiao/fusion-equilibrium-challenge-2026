#!/bin/bash
# Build the two pinned environments from the lock files (hash-checked) and fetch the pinned starter kit.
#   bash env/build_env.sh            -> src/.venv (PyTorch, CUDA 12.4), src/.venv_gs (FreeGS, CPU), src/.vendor_scorer (scikit-image 0.25.2)
# Python 3.11 is required (the lock files were resolved for CPython 3.11, manylinux_2_28, aarch64 and x86_64).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); WS=$REPO/src
PYBIN=${PYBIN:-python3.11}
ARCH=$(uname -m); case "$ARCH" in aarch64|arm64) ARCH=aarch64;; x86_64|amd64) ARCH=x86_64;; *) echo "unsupported architecture $ARCH"; exit 1;; esac
"$PYBIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || { echo "Python 3.11 required"; exit 1; }
mk() {  # name lockfile
  local v=$WS/$1
  [ -x "$v/bin/python" ] && [ "${FORCE:-0}" != 1 ] && { echo "$v exists (FORCE=1 to rebuild)"; return 0; }
  rm -rf "$v"; "$PYBIN" -m venv "$v"
  "$v/bin/python" -m pip install --quiet --upgrade "pip==26.2.1"
  "$v/bin/python" -m pip install --require-hashes --no-deps --only-binary=:all: -r "$REPO/env/$2"
  "$v/bin/python" -m pip check || true
  "$v/bin/python" -m pip freeze > "$v/freeze.txt"
}
mk .venv requirements-ml.$ARCH.lock
mk .venv_gs requirements-gs.$ARCH.lock
# scorer overlay for the MAST map corrector: scikit-image 0.25.2 (+ lazy-loader) as a target directory in front of the ML environment
if [ ! -d "$WS/.vendor_scorer/skimage" ] || [ "${FORCE:-0}" = 1 ]; then
  rm -rf "$WS/.vendor_scorer"
  "$WS/.venv/bin/python" -m pip install --quiet --require-hashes --no-deps --only-binary=:all: --target "$WS/.vendor_scorer" -r "$REPO/env/requirements-scorer.lock"
fi
bash "$REPO/tools/get_starter.sh"
"$WS/.venv/bin/python" - <<'PY'
import torch, numpy, scipy, sys
print("ml env: python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "numpy", numpy.__version__, "scipy", scipy.__version__)
PY
"$WS/.venv_gs/bin/python" -c 'import freegs, numpy, scipy, sys; print("gs env: python", sys.version.split()[0], "freegs", freegs.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)'
PYTHONPATH="$WS/.vendor_scorer" "$WS/.venv/bin/python" -c 'import skimage, sys; print("scorer overlay: scikit-image", skimage.__version__, "from", "vendor" if ".vendor_scorer" in skimage.__file__ else "ELSEWHERE")'
echo ENV_OK
