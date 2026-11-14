#!/bin/bash
# Place the submitted weights and frozen statistics under $SCR (paths as in the manifests) and verify every sha256.
#   bash tools/fetch_weights.sh hf  <hf_model_repo_id> [revision]    Hugging Face Hub model repository (files under weights/)
#   bash tools/fetch_weights.sh url <https url of weights.tar>       release asset / archive (tar with the manifest paths under weights/)
#   bash tools/fetch_weights.sh dir <local directory>                 local copy with the manifest paths
# Manifests (MANIFESTS="a b ..." to choose): weights/MANIFEST.sha256 (DIII-D model, 49 files), weights/MANIFEST_CH1_HC10X24.sha256
# (the 24 corrector heads of the DIII-D profile hc10x24; own archive weights_hc10x24.tar in url mode), weights/MANIFEST_CH2.sha256
# (shipped learned modules of the MAST pipeline, if any). Default: the manifests of the selected DIII-D profile (CH1_PROFILE, else
# ch1/profiles/DEFAULT: for hc10x24 the first two) + weights/MANIFEST_CH2.sha256 - i.e. everything the DEFAULT profiles need. In url
# mode one archive holds only some of them: give MANIFESTS per archive (e.g. MANIFESTS="weights/MANIFEST.sha256 weights/MANIFEST_CH2.sha256"
# for the bundle, MANIFESTS=weights/MANIFEST_CH1_HC10X24.sha256 for weights_hc10x24.tar). WEIGHTS_OK is printed only if every file of
# every manifest is present with its size and sha256. The training prerequisites (generated data, weights/GENERATED.sha256) are
# fetched the same way with MANIFESTS=weights/GENERATED.sha256 and the generated-data archive.
set -euo pipefail
. "$(dirname "$0")/../env.sh"
MODE=${1:?hf|url|dir}; SRC=${2:?source}; REV=${3:-main}
if [ -z "${MANIFESTS:-}" ]; then
  CHM=$( set +u; . "$FEC_REPO/ch1/common.sh" >/dev/null 2>&1; echo "$CH1_MANIFESTS" )
  [ -n "$CHM" ] || { echo "cannot read the manifests of the DIII-D profile ${CH1_PROFILE:-in ch1/profiles/DEFAULT}"; exit 2; }
  MANIFESTS="$CHM weights/MANIFEST_CH2.sha256"
fi
echo "manifests: $MANIFESTS"
PY=$WS/.venv/bin/python
case "$MODE" in
  hf)  "$PY" - "$SRC" "$REV" "$SCR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
p = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["weights/**"], local_dir=dst + "/.weights_dl")
print("downloaded to", p)
PY
       FROM=$SCR/.weights_dl/weights;;
  url) mkdir -p "$SCR/.weights_dl"; curl -fsSL --retry 5 -o "$SCR/.weights_dl/weights.tar" "$SRC"
       [ -z "${WEIGHTS_TAR_SHA256:-}" ] || echo "$WEIGHTS_TAR_SHA256  $SCR/.weights_dl/weights.tar" | sha256sum -c -
       tar -xf "$SCR/.weights_dl/weights.tar" -C "$SCR/.weights_dl"; rm -f "$SCR/.weights_dl/weights.tar"
       FROM=$SCR/.weights_dl/weights;;
  dir) FROM=$SRC;;
  *) echo "mode: hf | url | dir"; exit 1;;
esac
for m in $MANIFESTS; do
  [ -f "$FEC_REPO/$m" ] || { echo "no manifest $m"; continue; }
  grep -qv '^\s*\(#\|$\)' "$FEC_REPO/$m" || { echo "manifest $m: empty (nothing to fetch)"; continue; }
  "$PY" "$FEC_REPO/tools/verify_manifest.py" "$FEC_REPO/$m" "$SCR" --from "$FROM" || { echo "WEIGHTS_FAILED: $m"; exit 1; }
done
echo WEIGHTS_OK
