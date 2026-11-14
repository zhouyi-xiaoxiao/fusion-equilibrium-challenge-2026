# DIII-D profile "fccx" = recipe FcCX. This predict path writes diii_d_public_test.npz md5 0b13e8ef…; the FcCX file of the Development
# submissions of 2026-10-12 / 13 (md5 863586e8…) is the same recipe and differs from it in 3 of 193,359 frames (order of the two final
# guard steps; PROVENANCE.md section 2.7). Base map: mirror-canonical FM (11 nets) + TW (4 seeds), scalars SC (17 members); corrector: direct heads on the
# canonical FEATURE-V2.1 rows (residual MLP 768 x 4, 6 seeds), frozen rob constants, robmi Gauss-Newton step, l_i window guard.
CH1_DESC="FcCX (corrector heads: FEATURE-V2.1 rows, width 768 x 4 blocks, 6 seeds)"
HEADS=v21
CH1_MANIFESTS="weights/MANIFEST.sha256"
REF_MD5=0b13e8ef8df05634a0a3fc4c426455dd            # diii_d_public_test.npz written by this profile's predict path
