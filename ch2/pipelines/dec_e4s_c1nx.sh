# MAST pipeline profile "dec_e4s_c1nx" = DEC_E4s_c1nx_k50 (mast_public_test.npz md5 006ed728…, Development submission #17):
#   1. the 0.5 / 0.5 blend of syn_xa_p7li50 (network on our own synthetic equilibria + INV_P7li),
#   2. post stage "dec" (ch2/pipelines/dec_e4s.inc): the start-up frames (label-free regime rule) re-decoded toward per-functional targets
#      that combine the network's and the inverse's published functionals (DIII-D source weights + synthetic-MAST weights, support gate);
#      every other frame keeps the blend map,
#   3. post stage "corr" (ch2/pipelines/corr_b50.inc): the DIII-D-trained control-point corrector, setting c1nx (boundary x 0.5, l_i x 1,
#      no axis term, zero mean over the corpus), applied to the decoded maps; feature centring and zero mean recomputed on them.
# q95 / beta_N = those of syn_xa_p7li50, bit for bit. Stages: those of syn_xa_p7li50, post synblend + dec + corr.
. "$FEC_REPO/ch2/pipelines/syn_xa_p7li50.sh"
PIPE_DESC="syn_xa_p7li50 + start-up decoder (E4) + DIII-D-trained map corrector, setting c1nx (boundary x 0.5, l_i x 1, no axis, zero mean)"
. "$FEC_REPO/ch2/pipelines/dec_e4s.inc"
CORR_FLAGS=(--outs dRt,dZt,dRb,dZb,dRi,dRo --li --debias --alphas dRt=0.5,dZt=0.5,dRb=0.5,dZb=0.5,dRi=0.5,dRo=0.5)
. "$FEC_REPO/ch2/pipelines/corr_b50.inc"
REF_MD5=006ed7289f53e51c9602e8326b75c3c4
