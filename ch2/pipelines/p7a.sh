# MAST pipeline profile "p7a" = INV_P7a_k50 (Development submission #10, 2026-10-12; mast_public_test.npz md5 1915e1d3…).
# Free-boundary Grad-Shafranov kinetic inverse: G1 soft Thomson-edge closure, solenoid factor 0.5, fitted vector
# theta = [dr, lcp, b1, b2] (no shaping-field degree of freedom), FF' family "fe", priors from a DIII-D label fit.
# q95 = contour integral x C_q95, beta_N = c_betaN x electron beta_N (both constants from DIII-D, ch2/constants/p7a).
PIPE_DESC="kinetic inverse INV_P7a (theta = dr, lcp, b1, b2; solenoid factor 0.5)"
TAG=p7a
SOLVER_DIR=$WS/stage11_run/g_restrict
SOLVER=run_inv7.py
SOLVE_ARGS=(--cfg "$WS/stage11_run/ch2_G1/configs/cfg_soft_k50.json" --cfg "$WS/stage11_run/g_restrict/configs/inv7_base.json" --set 'params=["dr","lcp","b1","b2"]')
PCAL_ARGS=(--set "stress={}")                       # DIII-D calibration run: clean MAST-like pseudo-horizontal Thomson chord
PACKER=$WS/stage11_run/e_mast_inv6/pack_inv6.py
POST=()
REF_MD5=1915e1d3b5d50a1c7d86e0b4c55a931f            # mast_public_test.npz of this profile (Development phase)
# label fit behind the prior means / widths in g_restrict/configs/inv7_base.json (ch2/calibrate_d3d.sh <dir> <name> labelfit)
labelfit() {  # <diii_d_train dir> <out dir> <holdout spec>
  ( cd "$INV" && $PY run_inv.py --machine d3d --mode labelfit --data-dir "$1" --exclude-holdout "$3" --stride 80 --wall "$GEOM/d3d_limiter_mask.npz" \
      --cfg "$G1/configs/cfg_soft_k50.json" --cfg "$INV/configs/inv_lf.json" --set 'basis="fe"' --tag lf_fe --out "$2" --frames-per-shot 8 --procs "$NPROC" 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-1500
    $PY inv_summary.py "$2" lf_fe )
}
