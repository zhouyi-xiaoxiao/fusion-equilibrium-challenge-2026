# MAST pipeline profile "p7c" = INV_P7c_k50 (mast_public_test.npz md5 b576340d…; rebuilt from scratch to the same md5 by the
# rule-3 audit of 2026-10-12). INV_P7a with a different FF' / pressure profile layer: cubic FF' family with a fitted edge term
# (theta = [dr, lcp, b1, b2, be]), DIII-D-label-fitted prior means, pressure-gradient shape (1-u)(1+s u) with s = 0.5,
# conservative cut-cell source quadrature. q95 / beta_N constants re-derived on this pipeline's own DIII-D maps (ch2/constants/p7c).
PIPE_DESC="kinetic inverse INV_P7c (cubic FF' family with edge term, cut-cell quadrature)"
TAG=P7c
SOLVER_DIR=$WS/stage11_run/g_profile
SOLVER=run_p7.py
SOLVE_ARGS=(--cfg "$WS/stage11_run/ch2_G1/configs/cfg_soft_k50.json" --cfg "$WS/stage11_run/g_profile/configs/p7_base.json" --cfg "$WS/stage11_run/g_profile/configs/p7c_final.json")
PCAL_ARGS=(--cfg "$WS/stage11_run/g_profile/configs/st_clean.json")
PACKER=$WS/stage11_run/g_profile/pack_p7.py
POST=()
REF_MD5=b576340dc3ed986be0144f56a91b6732
# label fit behind the prior means of p7c_final.json (lcp0 = ln median c_p, b0 = medians of b1 / b2, be0 = median of be):
# the cubic family with wide priors and s = 0.5 fitted to the DIII-D label flux maps of 172 TRAIN shots x 8 frames
# (every 40th train shot, holdout and the 60 shots of splits/d3d_lf_disjoint60.txt excluded), then g_profile/mk_priors.py.
labelfit() {  # <diii_d_train dir> <out dir> <holdout spec>
  ( cd "$SOLVER_DIR" && $PY run_p7.py --machine d3d --mode labelfit --data-dir "$1" --exclude-holdout "$3" --stride 40 \
      --disjoint-from "$FEC_REPO/splits/d3d_lf_disjoint60.txt" --wall "$GEOM/d3d_limiter_mask.npz" \
      --cfg "$WS/stage11_run/ch2_G1/configs/cfg_soft_k50.json" --cfg "$WS/stage11_run/g_profile/configs/p7_base.json" \
      --cfg "$WS/stage11_run/g_profile/configs/p7_lf.json" --set lf_pe=pseudo --cfg "$FEC_REPO/ch2/pipelines/p7c_labelfit.json" \
      --tag lf_cubS_s05 --out "$2" --frames-per-shot 8 --procs "$NPROC" 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-1200 )
  $PYS "$WS/stage11_run/g_profile/mk_priors.py" "$2" lf_cubS_s05 cub "$2/cfg" | cut -c1-600
  $PYS - "$2/cfg/fam_lf_cubS_s05.json" "$WS/stage11_run/g_profile/configs/p7c_final.json" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for k in ("lcp0", "b0", "be0"):
    print(f"COMPARE {k:5s} new {a[k]!r} shipped {b[k]!r} {'IDENTICAL' if a[k] == b[k] else 'DIFFERENT'}")
PY
}
