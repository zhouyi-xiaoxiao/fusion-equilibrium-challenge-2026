# DIII-D profile "xhc" = hc10x24 with ONE change: the base map's 11 FM members and 4 TW seeds are the label-polygon contour fine-tune
# of the hc10x24 base members (training stage FT, ch1/train_xhc_ft.sh; METHODS.md section "Profile xhc"). Corrector heads (v2.2 c10,
# 24 seeds), frozen rob constants, Jacobian of the base (recomputed by the predict path for the new map), robmi step, l_i guard,
# final step, writer and gate are those of hc10x24; q95 / beta_N come from the same SC members (bit-identical to hc10x24).
# The predict path loads its FM / TW / SC files from BASE_ROOT (default: the xhc root of weights/MANIFEST_CH1_XHC.sha256, which also
# holds copies of the 17 unchanged SC files). Training: bash ch1/train_xhc.sh <MODEL_ROOT> <diii_d_train dir>.
CH1_DESC="xhc (hc10x24 + contour fine-tune of the base FM / TW members: label polygon, share 5, 40 epochs)"
HEADS=v22c10
HEAD22_ROOT=${HEAD22_ROOT:-$SCR/stage11_ch1/l_corr/heads22/full}
HEAD22_FEAT=${HEAD22_FEAT:-$SCR/stage11_ch1/l_corr/feat22_all}     # training cache of the heads (ch1/build_feat22.py; training only)
FEAT22_X_SHA256=c59bcd69b4fae413d0ca79b21f53eee036e9397da1b4504e561c1f6c681081bf   # train/X.npy of that cache in the development run
export BASE_ROOT=${BASE_ROOT:-$SCR/stage11_ch1/xhc/full}
CH1_MANIFESTS="weights/MANIFEST.sha256 weights/MANIFEST_CH1_HC10X24.sha256 weights/MANIFEST_CH1_XHC.sha256"
REF_MD5=""            # no Development-phase file of this profile (its package-path public-test md5 is in ch1/profiles/README.md)
