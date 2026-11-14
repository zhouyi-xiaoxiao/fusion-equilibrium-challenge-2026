# MAST pipeline profile "int_b50gld" = INT_B50gld_k50 (mast_public_test.npz md5 af5e3a7e…): the maps of syn_xa_p7li50 corrected by the
# DIII-D-trained control-point corrector (ch2/pipelines/corr_b50.inc), setting "gld": zero mean on the corpus, every output at strength 1
# (axis, four boundary control points, l_i re-map) - the strength the DIII-D holdout chooses. q95 / beta_N = those of syn_xa_p7li50.
# Stages: those of syn_xa_p7li50, post synblend + corr.
. "$FEC_REPO/ch2/pipelines/syn_xa_p7li50.sh"
PIPE_DESC="syn_xa_p7li50 + DIII-D-trained map corrector, setting gld (all outputs x 1, zero mean)"
CORR_FLAGS=(--li --debias)
. "$FEC_REPO/ch2/pipelines/corr_b50.inc"
REF_MD5=af5e3a7e5fd6b12c7daaf737355be1db
