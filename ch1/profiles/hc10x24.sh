# DIII-D profile "hc10x24" (diii_d_public_test.npz md5 0c9beef8…, Development submission of 2026-10-13). FcCX with ONE change: the
# corrector's direct heads are trained on the canonical FEATURE-V2.2 rows (the rows the base already reads: guarded Ip-clock rule for
# old-DAQ shots) with width 1024, 8 residual blocks, dropout 0.15 and 24 seeds (4 groups c10a..d x 6 seeds); targets, frozen constants,
# Jacobian, robmi step, l_i guard, writer and gate are those of FcCX. The 6 FEATURE-V2.1 head checkpoints are not used.
# Weights: weights/MANIFEST_CH1_HC10X24.sha256 (24 files) in addition to weights/MANIFEST.sha256.
CH1_DESC="hc10x24 (corrector heads: FEATURE-V2.2 rows, width 1024 x 8 blocks, dropout 0.15, 24 seeds)"
HEADS=v22c10
HEAD22_ROOT=${HEAD22_ROOT:-$SCR/stage11_ch1/l_corr/heads22/full}
HEAD22_FEAT=${HEAD22_FEAT:-$SCR/stage11_ch1/l_corr/feat22_all}     # training cache of the heads (ch1/build_feat22.py; training only)
FEAT22_X_SHA256=c59bcd69b4fae413d0ca79b21f53eee036e9397da1b4504e561c1f6c681081bf   # train/X.npy of that cache in the development run
CH1_MANIFESTS="weights/MANIFEST.sha256 weights/MANIFEST_CH1_HC10X24.sha256"
REF_MD5=0c9beef87ee0b8ac63c1f628f7ccdf2c            # diii_d_public_test.npz of this profile (Development phase)
