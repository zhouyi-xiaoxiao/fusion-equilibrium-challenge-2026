# DIII-D model profiles

`ch1/predict.sh` and `ch1/train.sh` run the profile named by `CH1_PROFILE` (`ch1/profiles/<name>.sh`); without it the name in
`ch1/profiles/DEFAULT` is used. A profile fixes which corrector heads the predict path loads and the training path trains; every
other stage (features, canonical sign step, base map FM + TW, scalars SC, blend, Jacobian, frozen constants, robmi step, l_i guard,
writer, gate) is shared.

| profile | corrector heads | weights | public-test file of this predict path (`diii_d_public_test.npz`) |
|---|---|---|---|
| `fccx` | canonical FEATURE-V2.1 rows, residual MLP 768 × 4, 6 seeds (`HEADS=v21`) | `weights/MANIFEST.sha256` (49 files) | md5 `0b13e8ef8df05634a0a3fc4c426455dd` (the submitted Development file of this recipe, md5 `863586e8…`, differs in 3 frames: order of the two final guard steps) |
| `hc10x24` | canonical FEATURE-V2.2 rows (the base's own rows), residual MLP 1024 × 8, dropout 0.15, 24 seeds (`HEADS=v22c10`) | `weights/MANIFEST.sha256` + `weights/MANIFEST_CH1_HC10X24.sha256` (24 files, 1.69 GB) | md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c` |
| `xhc` | as `hc10x24` (`HEADS=v22c10`); base map from the contour fine-tuned FM / TW members (`BASE_ROOT` = the xhc root) | `weights/MANIFEST.sha256` + `weights/MANIFEST_CH1_HC10X24.sha256` + `weights/MANIFEST_CH1_XHC.sha256` (32 files: 11 FM + 4 TW fine-tuned, 17 SC copies) | md5 `XHC_PUBLIC_MD5` (package path; not a Development submission) |

`xhc` differs from `hc10x24` only in the base map's FM and TW members: they are the hc10x24 base members after the label-polygon
contour fine-tune (training stage FT, `ch1/train_xhc_ft.sh`; METHODS.md). Heads, frozen constants, the Gauss-Newton step and guards
are those of hc10x24, the Jacobian is recomputed for the new base by the predict path as for every input, and q95 / beta_N (SC
members, unchanged) are bit-identical to hc10x24. The profile sets `BASE_ROOT` (default `$SCR/stage11_ch1/xhc/full`); its training
entrypoint is `ch1/train_xhc.sh <MODEL_ROOT> <diii_d_train dir>` (base + heads through `ch1/train.sh`, then the FT stage).

`hc10x24` differs from `fccx` only in the heads: same targets (the 7 scorer-derived scalars + 4 LCFS extrema), same frozen rob
constants, same Jacobian, same Gauss-Newton step and guards. Its heads were selected on DIII-D validation folds (METHODS.md
section 2); q95 / beta_N of the two files are bit-identical, the flux maps differ by a median 4.6e-4 of the per-shot flux range.

## Interface

A profile is a bash file sourced by `ch1/common.sh` after `env.sh`:

| variable | required | meaning |
|---|---|---|
| `HEADS` | yes | `v21` or `v22c10`: switch of `src/stage11_run/ch1_package_final/predict.sh` / `train.sh` |
| `HEAD22_ROOT` | for `v22c10` | directory of the 24 head checkpoints `ck_c10{a,b,c,d}_s*.pt` (default: the manifest path under the work directory) |
| `HEAD22_FEAT`, `FEAT22_X_SHA256` | for training `v22c10` | head-training cache built by `ch1/build_feat22.py` from files of `weights/GENERATED.sha256`, and the sha256 of its `train/X.npy` |
| `CH1_MANIFESTS` | no | weight manifests the profile needs (for `tools/fetch_weights.sh`, `MANIFESTS=...`) |
| `REF_MD5` | no | md5 of the profile's Development-phase file (printed for comparison only) |
| `CH1_DESC` | no | one-line description (printed) |

## What a correct run prints

`ch1/predict.sh` hands the profile's `HEADS` to the inner script (`src/stage11_run/ch1_package_final/predict.sh`, whose own default
is the `hc10x24` heads — do not call it directly). The log of a finished run contains, in this order:

```
=== ch1 predict profile=<name> (...) HEADS=<v21|v22c10>
=== predict ... HEADS=<v21|v22c10>
--- [..s] heads v2.2 c10 (24 seeds)            (profile hc10x24 only)
G6 zero derivation failures: PASS
=== GATE PASS: ...
HEADS_CHECK OK: profile <name>, HEADS=..., v2.2 head groups predicted: <0|4>
profile <name>: md5 <md5 of the file>; Development-phase reference for diii_d_public_test: <REF_MD5>   (only when the output name is diii_d_public_test.npz)
ALL_DONE ...
```

`HEADS_CHECK FAILED` means the file was made with the other profile's heads; `PREDICT_INCOMPLETE` (exit status 1) means the inner
script ended without a finished file (for example because a file of the checkout was replaced while the job was running). With
`USE_SLURM=1` the inner script submits itself — the package's own copy — as one job and the same lines appear in the job log.

## Commands

```bash
# weights of hc10x24 (in addition to the base manifest)
MANIFESTS=weights/MANIFEST_CH1_HC10X24.sha256 bash tools/fetch_weights.sh url <https url of weights_hc10x24.tar>     # or: hf <repo> <rev>
# predict
CH1_PROFILE=hc10x24 bash ch1/predict.sh <d3d_shot_dir> <run_name> <stream_order.txt> <output file name>
# retrain (base 4 GPUs + 4 head jobs of 1 GPU under SLURM; NGPU GPUs locally) and compare with the manifest weights
CH1_PROFILE=hc10x24 USE_SLURM=1 CMP=1 bash ch1/train.sh <MODEL_ROOT>
CH1_PROFILE=hc10x24 BASE_ROOT=<MODEL_ROOT>/base HEAD_ROOT=<MODEL_ROOT>/heads HEAD22_ROOT=<MODEL_ROOT>/heads22 bash ch1/predict.sh <d3d_shot_dir> <run_name>
# profile xhc: weights (in addition to the two manifests above), retrain (base + heads + FT stage), predict from the retrained root
MANIFESTS=weights/MANIFEST_CH1_XHC.sha256 bash tools/fetch_weights.sh <hf|url|dir> <source>
USE_SLURM=1 CMP=1 bash ch1/train_xhc.sh <MODEL_ROOT> <diii_d_train dir>
CH1_PROFILE=xhc BASE_ROOT=<MODEL_ROOT>/xhc HEAD_ROOT=<MODEL_ROOT>/heads HEAD22_ROOT=<MODEL_ROOT>/heads22 bash ch1/predict.sh <d3d_shot_dir> <run_name>
```

A new DIII-D profile that changes more than the heads needs its own switch in the two scripts under
`src/stage11_run/ch1_package_final/`, its weights in a manifest of its own, and a clean-room run that reproduces its
Development-phase file (md5) before it is used for a Final submission.
