# MAST pipeline profile "int_b50c1nx" = INT_B50c1nx_k50 (mast_public_test.npz md5 275f4f59…, Development submission #15): the maps of
# syn_xa_p7li50 (0.5 x network + 0.5 x INV_P7li) corrected by the DIII-D-trained control-point corrector (ch2/pipelines/corr_b50.inc),
# setting "c1nx": zero mean on the corpus, boundary control points (top, bottom, inboard, outboard) x 0.5, l_i re-map x 1, no axis term.
# q95 / beta_N = those of syn_xa_p7li50, bit for bit. Strength pattern fixed on DIII-D (holdout + untouched extra set) and on label-free
# MAST signals (Thomson edge and symmetry centre; METHODS.md section 3) - the transfer-guarded setting of the corrector's one strength knob.
# Stages: those of syn_xa_p7li50, post synblend + corr.
. "$FEC_REPO/ch2/pipelines/syn_xa_p7li50.sh"
PIPE_DESC="syn_xa_p7li50 + DIII-D-trained map corrector, setting c1nx (boundary x 0.5, l_i x 1, no axis, zero mean)"
CORR_FLAGS=(--outs dRt,dZt,dRb,dZb,dRi,dRo --li --debias --alphas dRt=0.5,dZt=0.5,dRb=0.5,dZb=0.5,dRi=0.5,dRo=0.5)
. "$FEC_REPO/ch2/pipelines/corr_b50.inc"
REF_MD5=275f4f59fdecdef368cdff35fda09ad5
