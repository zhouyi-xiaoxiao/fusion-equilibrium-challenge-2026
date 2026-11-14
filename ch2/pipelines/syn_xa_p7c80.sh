# MAST pipeline profile "syn_xa_p7c80" = SYN_XA_P7c80_k50 (mast_public_test.npz md5 c46183b8…): map blend of 0.8 x the network "XA"
# (ch2/pipelines/syn_net.inc) and 0.2 x the INV_P7c solution (profile p7c); scalars blended after each component's DIII-D calibration.
# Weight 0.8 chosen on the 150 DIII-D holdout shots (all frames, best under the stressed Thomson condition), confirmed on an untouched set.
. "$FEC_REPO/ch2/pipelines/p7c.sh"
CONST_PIPE=p7c
PIPE_DESC="0.8 x network on own synthetic equilibria + 0.2 x INV_P7c (map blend)"
SYN_W=0.8
. "$FEC_REPO/ch2/pipelines/syn_net.inc"
REF_MD5=c46183b816983ea499bafba56cdb5d6b
