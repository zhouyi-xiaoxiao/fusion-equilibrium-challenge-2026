# Ch1 (DIII-D model) profile selection. Sourced by ch1/predict.sh and ch1/train.sh after env.sh.
# CH1_PROFILE=<name> loads ch1/profiles/<name>.sh (default: the name in ch1/profiles/DEFAULT). A profile sets HEADS (which corrector
# heads the predict path loads / the training path trains), the weight manifests it needs and the md5 of its Development-phase file.
CH1_PROFILE=${CH1_PROFILE:-$(tr -d ' \n' < "$FEC_REPO/ch1/profiles/DEFAULT")}
export CH1_PROFILE
_P=$FEC_REPO/ch1/profiles/$CH1_PROFILE.sh
[ -f "$_P" ] || { echo "unknown DIII-D profile '$CH1_PROFILE' (profiles: $(cd "$FEC_REPO/ch1/profiles" && ls *.sh | sed 's/\.sh$//' | tr '\n' ' '))"; exit 2; }
# HEAD22_ROOT / HEAD22_FEAT given in the environment override the profile's defaults (the profile reads them as ${VAR:-default});
# every other profile variable is set by the profile alone.
_ENV_HEAD22_ROOT=${HEAD22_ROOT:-}; _ENV_HEAD22_FEAT=${HEAD22_FEAT:-}
HEADS=""; HEAD22_ROOT=$_ENV_HEAD22_ROOT; HEAD22_FEAT=$_ENV_HEAD22_FEAT; CH1_MANIFESTS="weights/MANIFEST.sha256"; REF_MD5=""; CH1_DESC=""
. "$_P"
[ -n "$HEADS" ] || { echo "profile $_P must set HEADS"; exit 2; }
export HEADS
[ -z "$HEAD22_ROOT" ] || export HEAD22_ROOT
[ -z "$HEAD22_FEAT" ] || export HEAD22_FEAT
if [ -n "$_ENV_HEAD22_ROOT" ]; then
  if [ "$HEADS" = v22c10 ]; then echo "NOTE: HEAD22_ROOT from the environment overrides the profile default: $HEAD22_ROOT"
  else echo "NOTE: HEAD22_ROOT is set in the environment but profile $CH1_PROFILE (HEADS=$HEADS) does not use the v2.2 heads"; fi
fi
[ -z "$_ENV_HEAD22_FEAT" ] || echo "NOTE: HEAD22_FEAT from the environment overrides the profile default: $HEAD22_FEAT"
