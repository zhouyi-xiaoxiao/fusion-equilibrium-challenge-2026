# MAST pipeline profile "syn_xa_p7li50" = SYN_XA_P7li50_k50 (mast_public_test.npz md5 11d94769…): map blend of
#   0.5 x amortised network "XA" trained on our own synthetic free-boundary equilibria (driven by the input columns of the shots being
#   predicted; ch2/pipelines/syn_net.inc) and 0.5 x the INV_P7li solution (profile p7li); scalars blended after each component's own
#   DIII-D calibration. Weight 0.5 chosen on the 150 DIII-D holdout shots (all frames), confirmed on an untouched 150-shot DIII-D set.
# Stages: solve (p7li) | synreal syngen (CPU, shardable) | syntrain (GPU) | merge scal pack (p7li file) | post synblend | validate.
. "$FEC_REPO/ch2/pipelines/p7li.sh"
CONST_PIPE=p7li                                     # q95 / beta_N constants of the inverse component
PIPE_DESC="0.5 x network on own synthetic equilibria + 0.5 x INV_P7li (map blend)"
SYN_W=0.5
. "$FEC_REPO/ch2/pipelines/syn_net.inc"
REF_MD5=11d94769bb5cc5cb9c0b4b11accd6c87
