# MAST pipeline profile "dec_e4s_c1nx_a1w5" = DEC_E4s_c1nx_A1w5_k50 (mast_public_test.npz md5 4d66f8fc…): the maps of dec_e4s_c1nx bit for bit,
# q95 by rule A1 (follows the corrector's map change), beta_N by rule B7w5 (5-frame median); post stage "a1w5" (ch2/pipelines/scal_a1w5.inc).
# Stages: those of dec_e4s_c1nx, post synblend + dec + corr + a1w5.
. "$FEC_REPO/ch2/pipelines/dec_e4s_c1nx.sh"
PIPE_DESC="dec_e4s_c1nx + scalar rules A1 (q95 follows the corrected map) and B7w5 (beta_N 5-frame median)"
. "$FEC_REPO/ch2/pipelines/scal_a1w5.inc"
REF_MD5=4d66f8fc0bf014dd329b717bf8695d20
