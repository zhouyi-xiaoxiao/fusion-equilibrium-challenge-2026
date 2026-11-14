# Fusion Equilibrium Challenge (NeurIPS 2026) — reproduction package

Code, pinned environments, weights manifest and entrypoints that regenerate the two submission files of our entry:

| file | machine | method | entrypoints |
|---|---|---|---|
| `diii_d_*_test.npz` | DIII-D (Challenge 1) | learned: frame MLPs + temporal window model on PCA coefficients of the flux map, scalar models for q95 / β_N, a scalar-consistency correction of the map; profiles (`ch1/profiles/`): `fccx` and `hc10x24` differ in the correction heads, `xhc` is `hc10x24` with the base members contour fine-tuned | `ch1/train.sh` (`ch1/train_xhc.sh` for `xhc`), `ch1/predict.sh` |
| `mast_*_test.npz` | MAST (Challenge 2, zero-shot) | physics: free-boundary Grad–Shafranov kinetic inverse from coil currents and Thomson profiles, optionally blended with a network trained on synthetic equilibria of the same forward model, its start-up frames re-decoded per functional and corrected by a DIII-D-trained map corrector; selected by a pipeline profile (`ch2/pipelines/`); constants and learned modules from DIII-D labels and our own synthetic equilibria only, no MAST label | `ch2/calibrate_d3d.sh`, `ch2/train_syn_d3d.sh`, `ch2/train_corr_b50.sh`, `ch2/train_dec_e4s.sh`, `ch2/train_mast_net.sh` (network trained on the inputs being predicted), `ch2/predict_mast.sh` |

Read `PROVENANCE.md` for the per-machine data-provenance statement and the declared compute, `METHODS.md` for the method,
`RUNBOOK_FINAL.md` for the Final-phase procedure. Licence: MIT (`LICENSE`). The first-wall polygon in
`ch2/geometry/mast_limiter.json` is FAIR-MAST machine-description data (CC BY-SA 4.0, see `ch2/geometry/NOTICE`).

## Layout

```
env.sh                    sets WS (= src/) and SCR (= work directory, from FEC_WORK)
env/                      lock files (hash-pinned, CPython 3.11, aarch64 and x86_64) and build_env.sh
ch1/predict.sh            DIII-D inference: parquet shots -> .npz          ch1/train.sh   from-scratch training of the weight files
ch1/profiles/             DIII-D profiles (CH1_PROFILE=<name>; DEFAULT names the submitted one): fccx, hc10x24, xhc
ch1/train_xhc.sh          training entrypoint of profile xhc: ch1/train.sh, then the contour fine-tune stage ch1/train_xhc_ft.sh
                          (xhc_assemble.py builds the base root its predict path loads and compares it with the manifest)
ch1/build_feat22.py       training cache of the hc10x24 heads, built from files of weights/GENERATED.sha256
ch1/frozen/               small frozen statistics kept in the repository
ch2/predict_mast.sh       MAST inference: parquet shots -> .npz             ch2/calibrate_d3d.sh   DIII-D-derived constants
ch2/train_syn_d3d.sh      DIII-D twin of the synthetic-equilibrium network (its two scalar constants)
ch2/train_corr_b50.sh     DIII-D derivation of the map corrector of the int_b50* and dec_e4s_c1nx profiles (weights/MANIFEST_CH2.sha256)
ch2/train_dec_e4s.sh      derivation of the start-up decoder's weights and support table (dec_e4s_c1nx; on the runs of train_corr_b50.sh)
ch2/train_mast_net.sh     training entrypoint of the MAST network trained on the inputs being predicted (syn_*, int_*, dec_* profiles;
                          in a Final run: on the Final inputs); calls the unchanged ch2/predict_mast.sh
ch2/pipelines/            MAST pipeline profiles (PIPE=<name>; DEFAULT names the submitted one) and the profile interface
ch2/constants/<pipe>/     DIII-D-derived constants of each pipeline (C_q95, c_betaN; the l_i model of p7li; the network's constants)
ch2/geometry/             public first-wall polygons and the grid masks built from them (with sources)
splits/                   DIII-D shot lists: holdout (150 shots), label-fit exclusion, l_i model train / check lists and their
                          table order, exclusion lists of the network twin
weights/                  MANIFEST.sha256 (DIII-D model: 49 files, 1.21 GB), MANIFEST_CH1_HC10X24.sha256 (24 heads of profile
                          hc10x24, 1.69 GB), MANIFEST_CH1_XHC.sha256 (base root of profile xhc: 15 fine-tuned files + 17
                          unchanged SC files, 1.08 GB), MANIFEST_CH2.sha256 (shipped learned MAST modules; a syn_* network trained
                          by a Final run is added here), GENERATED.sha256 and GENERATED_XHC.sha256 (training prerequisites, see
                          docs/CH1_GENERATED_DATA.md)
tools/                    download_inputs.py, stream_order.py, fetch_weights.sh, verify_manifest.py, validate_final.py,
                          reorder_npz.py, publish_model.py, get_starter.sh, open_audit/ (optional file-read audit),
                          final_slurm.sh (RUNBOOK_FINAL.md section B as SLURM jobs; check / status)
src/                      the code the entrypoints call (module map below); the organisers' starter kit is fetched at a pinned
                          commit into src/starter and used unmodified (scorer, validator, data_fixes.py)
```

## Weights

This repository holds the code, the pinned environments and the sha256 manifests (`weights/*.sha256`), but not the trained
weights or the generated training data (about 21 GB). Both can be regenerated with the training entrypoints (section
"Training"); the scripts that produce the generated data are listed in `docs/CH1_GENERATED_DATA.md`. The weights shipped with
our Final submissions are available from the author on request; `tools/fetch_weights.sh dir <local copy>` places them and
checks every file against the manifests. Regenerated weights agree with the shipped ones to the extent stated in
`PROVENANCE.md` section 3.

## Quick start

Requirements: Linux, bash ≥ 4.4, CPython 3.11, for DIII-D inference one CUDA GPU (CUDA 12.4 wheels).

```bash
export FEC_WORK=/path/to/work            # >= 30 GB for inference, >= 150 GB for training
bash env/build_env.sh                    # two virtual environments from the lock files (pip --require-hashes) + starter kit
bash tools/fetch_weights.sh hf <model repo id> <commit>   # or: url <weights.tar URL> | dir <local copy>; verifies every sha256

# inputs (one parquet file per shot) and the stream order of each configuration
src/.venv/bin/python tools/download_inputs.py --repo Sophelio/fusion-equilibrium-challenge-final --config diii_d_private_test --out $FEC_WORK/in/d3d
src/.venv/bin/python tools/download_inputs.py --repo Sophelio/fusion-equilibrium-challenge-final --config mast_private_test   --out $FEC_WORK/in/mast
src/.venv/bin/python tools/stream_order.py --shot-dir $FEC_WORK/in/d3d  --repo Sophelio/fusion-equilibrium-challenge-final --config diii_d_private_test --out-dir $FEC_WORK/order/d3d
src/.venv/bin/python tools/stream_order.py --shot-dir $FEC_WORK/in/mast --repo Sophelio/fusion-equilibrium-challenge-final --config mast_private_test   --out-dir $FEC_WORK/order/mast

bash ch1/predict.sh       $FEC_WORK/in/d3d  final $FEC_WORK/order/d3d/stream_order.txt  diii_d_private_test.npz   # -> $FEC_WORK/stage11_ch1/final_path/runs/final/
# MAST with the published network of a Final submission (inference only; the file is listed in weights/MANIFEST_CH2.sha256 and placed
# in FEC_WORK by tools/fetch_weights.sh, e.g. $FEC_WORK/stage11_ch2/final/model_mCxa32.pt):
SYN_MODEL_FILE=$FEC_WORK/<path listed in weights/MANIFEST_CH2.sha256> \
bash ch2/predict_mast.sh  $FEC_WORK/in/mast final $FEC_WORK/order/mast/stream_order.txt mast_private_test.npz     # -> $FEC_WORK/ch2_runs/final/submission/
```

**MAST inference with the shipped network.** `SYN_MODEL_FILE=<published model>`, in a new run directory, makes the stage `syntrain`
copy that file into the run directory instead of training (a model already in the run directory is used as it is;
`ch2/pipelines/syn_net.inc`) and then predict every frame with it. Without `SYN_MODEL_FILE`,
`predict_mast.sh` trains the network again on the inputs being predicted: that is the training step (`ch2/train_mast_net.sh`,
section "Training"), not inference with the shipped weights. The synthetic set is still required: before it takes the given file,
the stage `syntrain` checks that the stage `syngen` has written the synthetic set of every shot (CPU, 333 core-h for 1,206 shots,
shardable) and that the stage `synreal` has written the network inputs; the single call above runs every stage in order (solve,
synreal, syngen, syntrain, merge, scal, pack, post, validate). On a cluster, split it as in `RUNBOOK_FINAL.md` B.3–B.5 with the
same `NSHARD` in every call and `SYN_MODEL_FILE` on the `syntrain` call. Pass the stream-order file of the configuration
(`tools/stream_order.py`), not `sorted`. The Final files were computed on aarch64 (Arm Neoverse-V2 cores, GH200 GPUs) with the
`aarch64` lock files, OpenBLAS choosing its kernel for that CPU (the package sets no kernel override such as `OPENBLAS_CORETYPE`;
the kernel of the Final run: `PROVENANCE.md` section 3) and the network step on the device recorded in `PROVENANCE.md` section 4
(one GH200 GPU unless the CPU fallback of `RUNBOOK_FINAL.md` was used); the published copy
holds the run's network arrays bit for bit (`tools/publish_model.py`), and `RUNBOOK_FINAL.md` B.7 prescribes a check that
this path reproduces the md5 of each Final file. For the MAST file of Final submissions 1002494 and 1006850 the check reproduced
the md5; it was run after the upload of 1002494, in two parts (19 October and 10 November 2026; `PROVENANCE.md` section 3). For the MAST file of 1003753, which is that file
after one post stage, it was not run separately (`PROVENANCE.md` header). On another BLAS kernel or architecture the inverse is not bit-identical (measured with the generic
ARMv8 kernel for `p7a` on DIII-D labels: ΔS = +0.00012; `PROVENANCE.md` section 3).

The DIII-D entrypoints run the profile named in `ch1/profiles/DEFAULT` (`CH1_PROFILE=<name>` for the other one; `hc10x24` also
needs `MANIFESTS=weights/MANIFEST_CH1_HC10X24.sha256 bash tools/fetch_weights.sh ...`).

`sorted` = shots in sorted file-name order, which is the stream order of the Development configurations;
`tools/stream_order.py` re-derives the order from the Hub for any configuration and both entrypoints accept the resulting list
instead of `sorted` (for a Final file always pass the order file, `RUNBOOK_FINAL.md` B.2). The MAST entrypoint runs the profile
named in `ch2/pipelines/DEFAULT` (`PIPE=<name>` for another one) and can be split into independent shards (`NSHARD=N SHARD=k`, then the finish stages; see its header and `RUNBOOK_FINAL.md`).
The `syn_*` profiles add CPU generation stages and one GPU stage that trains a network on synthetic equilibria driven by the
input columns of the shots being predicted (`ch2/pipelines/README.md`).

### Training

```bash
PIPE=<name> bash ch2/calibrate_d3d.sh <diii_d_train dir> calib pcal   # CPU only; prints COMPARE lines against ch2/constants/<name>
MANIFESTS=weights/GENERATED.sha256 bash tools/fetch_weights.sh <mode> <source>   # training prerequisites (generated data)
bash ch1/train.sh [MODEL_ROOT]                                        # FM 11 + TW 4 + SC 17 + 6 heads = 38 weight files (profile fccx)
BASE_ROOT=MODEL_ROOT/base HEAD_ROOT=MODEL_ROOT/heads bash ch1/predict.sh <shots> <name>
CH1_PROFILE=hc10x24 bash ch1/train.sh [MODEL_ROOT]                    # FM 11 + TW 4 + SC 17 + 24 heads (heads in MODEL_ROOT/heads22)
bash ch1/train_xhc.sh MODEL_ROOT <diii_d_train dir>                   # profile xhc: the above, then the contour fine-tune of FM / TW
CH1_PROFILE=xhc BASE_ROOT=MODEL_ROOT/xhc HEAD_ROOT=MODEL_ROOT/heads HEAD22_ROOT=MODEL_ROOT/heads22 bash ch1/predict.sh <shots> <name>
PIPE=p7li bash ch2/calibrate_d3d.sh <diii_d_train dir> calib pcal lifit   # + the l_i model of p7li (DIII-D labels)
bash ch2/train_syn_d3d.sh <diii_d_train dir> twin                     # constants of the syn_* network (CPU generation + 1 GPU, 4 min)
bash ch2/train_corr_b50.sh <diii_d_train dir> corr <twin>/models/model_Cxa32.pt   # map corrector of int_b50* (CPU only, about 460 core-h)
bash ch2/train_dec_e4s.sh <diii_d_train dir> corr                      # decoder weights of dec_e4s_c1nx from the same pool runs (CPU only)
bash ch2/train_mast_net.sh <mast shot dir> <run>                       # the network trained on the inputs being predicted (Final inputs in a
                                                                      # Final run): CPU generation 333 core-h + 1 GPU, about 3 min training
```

`ch2/train_mast_net.sh` is the training entrypoint of the Final-input MAST network (the transductive network of the `syn_*`,
`int_*` and `dec_*` profiles, among them the submitted `dec_e4s_c1nx`). It runs the steps `init` → `syngen` (every shard; `NSHARD`,
default 32, and `SHARD=k` as in `predict_mast.sh`) → `synreal` → `syntrain` of the profile in `ch2/pipelines/DEFAULT`
(`PIPE=<name>` for another `syn_*`, `int_*` or `dec_*` profile) through the unchanged `ch2/predict_mast.sh`, with the same calls as
the jobs of `tools/final_slurm.sh` (RUNBOOK_FINAL.md B.3 / B.4), and prints the model file
(`$SCR/ch2_runs/<run>/syn/models/model_mCxa32.pt`; `SCR` = `FEC_WORK` unless set, `env.sh`), its sha256 and the training wall time
of the trainer's log (`DEVICE=gpu|cpu`, `STEPS=…`, `DRYRUN=1`: see its header). It trains on one GPU, and that GPU time counts toward
the 24 GPU-hours with the rest of the training (`PROVENANCE.md` section 4). In a Final run the same calls run on the Final inputs
(in the Final run as jobs of `tools/final_slurm.sh`, from commit `82e50ff`, which predates this wrapper), and the shipped network of
each Final submission is the network that its Final run trained (published copy listed in `weights/MANIFEST_CH2.sha256`,
`RUNBOOK_FINAL.md` B.7). The inference stages of the same run directory (solve, then merge scal pack post validate with the
stream-order file) follow as in `RUNBOOK_FINAL.md` B.3–B.5, with the same `NSHARD` as the wrapper (default 32; `predict_mast.sh`
itself defaults to 1, which on a 32-shard run directory solves shard 0 only and stops at merge).

## Hardware, timings, determinism (measured)

All numbers are from one node type: 4 × NVIDIA GH200 (Grace Hopper, 120 GB GPU memory each, 288 Arm Neoverse-V2 cores per node),
aarch64, SLES 15 SP7, the `aarch64` lock files.

| step | resources | wall time |
|---|---|---|
| environments from the lock files | 16 cores | 3–6 min |
| Ch1 predict, 874 shots / 193,359 frames | 288 cores, 1 GPU used | 38 min (GPU busy < 2 min; the scalar-Jacobian step is 81 core-h) |
| Ch1 train, 38 weight files (`fccx`) | 4 GPUs 24 min (base) + 1 GPU 10 min (heads) | 1.75–1.79 GPU-h |
| Ch1 train, heads of `hc10x24` (24 seeds) | 4 jobs × 1 GPU, 18–28 min each | 1.1–1.4 GPU-h (instead of the 0.17 GPU-h of the `fccx` heads) |
| Ch2 predict (`p7a`), 1,206 shots / 69,075 frames | CPU only, about 200 core-h; shards are independent | about 15 min on 32 × 40 cores |
| Ch2 calibration `pcal` | CPU only, 144 cores | 5 min (10.4 core-h) |
| Ch2 `syn_*`: synthetic generation for 1,206 shots | CPU only, 333 core-h; shards are independent | 12 min on 32 × 72 cores |
| Ch2 `syn_*`: network training + prediction of 69,075 frames (`ch2/train_mast_net.sh`, step `syntrain`) | 1 GPU | 4 min (training 162–173 s) |
| Ch2 `syn_*`: DIII-D twin (generation 224 core-h, training 196 s on 1 GPU) | CPU + 1 GPU | 10 min on 12 nodes + 4 min |
| Ch2 `dec_e4s_c1nx`: finish after the network (pack, blend, start-up decoder, corrector, local check) | CPU only, 72 cores | 8.5 min (decoder stage 4 min; `dec_e4s_c1nx_a1w5`: + 3 min) |

- **Ch1** inference is bit-reproducible on this node type with the locked environment (the file regenerated from raw parquet
  has the md5 of the reference file). On other hardware expect float-level differences in the network outputs (measured CPU
  versus GPU: 5e-6 in PCA coefficients); the heads run in bf16 on the GPU.
- **Ch2** has no random element at inference. Every process is single-threaded BLAS, shots are independent and the result
  does not depend on the shard split or on the number of worker processes (identical md5 with 12 × 104 and 32 × 40 cores).
  The inverse is an iterative non-linear fit with discrete branch decisions, so a different CPU architecture or BLAS kernel
  changes last bits that can flip a branch on individual frames; the measured size of that effect is in `PROVENANCE.md`
  (section 3).
- Training is seeded and reproduces the shipped weights on this node type from the same generated data (34 of 38 files
  sha256-identical, the 4 temporal-model checkpoints tensor-identical with different serialisation metadata).
- The `syn_*` network is seeded as well: generation is seeded per shot (independent of the sharding) and the retrained network
  is tensor-identical on this node type, so the whole chain from parquet reproduces the profile's file (same md5).

## Module map (`src/stage11_run/`, names kept from development)

| directory | role |
|---|---|
| `ch1_package_final/` | Ch1 predict / train drivers, feature pass, blend, writer, stream-order check |
| `ch1_features/`, `e_ip_axis/` | input features v2.1 / v2.2 (coil currents, plasma current, Thomson profiles on the EFIT time base; Ip-clock rule) |
| `ch1d_fold2/` (`f2_build.py`, `vendor/`) | scaler helpers; the trainer copies used for the shipped weights (frame MLP, temporal, scalars) |
| `ch1_frame_mlp/`, `ch1_temporal/`, `ch1_scalars/` | predictors for the PCA coefficients of ψ (frame MLP, temporal window) and for q95 / β_N |
| `ch1_metric/` | metric loss (score-aligned quadratic form in PCA space) |
| `ch1_corrector/`, `ch1d_specialists/`, `ch1d_liloss/` | scalar-consistency correction (direct heads, Gauss–Newton step, metric prior), l_i window guard |
| `l_corr/` (`l_hp.py`) | trainer / predictor of the `hc10x24` heads (the `ch1_corrector` head code with width, depth and dropout as parameters) |
| `ch1_blend/`, `f_ch1_harden/` | output verification, final gate |
| `ch2_geometry/` | rule-3 process guard, Thomson geometry, public wall fetch + mask builder |
| `ch2_G1/` | free-boundary GS forward solver with Thomson-edge closure ("G1"), frame evaluation on DIII-D |
| `ch2d_inverse/`, `e_mast_inv6/`, `g_restrict/` | kinetic inverse (layers P5 → P6 → P7a; `g_restrict/run_inv7.py` is the solver of profile `p7a`) |
| `g_profile/` | profile layer of `p7c` (cubic FF′ family with edge term, cut-cell quadrature; `run_p7.py`, `pack_p7.py`, label-fit priors) |
| `j_temporal/` | temporal smoother of the plasma flux (post stage of `p7a_tsw05`) |
| `j_li_startup/` | l_i datum of `p7li`: feature / model code (`li_core.py`), table builder and fit (`li_table.py`, `li_fit.py`), solver driver `run_inv7l.py` (generated from `g_restrict/run_inv7.py` by `make_driver.py`) |
| `j_synth_net/` | network of the `syn_*` profiles: synthetic-equilibrium generator and input features (`syn_gen.py`, `syn_common.py`), trainer, predictor, blend and packer |
| `h_p2chan/` | P2 winding currents from the shipped input channels (used by the `syn_*` generator and features) |
| `k_corrector/` | map corrector of the `int_b50*` and `dec_e4s_c1nx` profiles: geometry table with the scorer's functionals (`ktab.py`), features (`kcommon.py`), fit (`kfit.py`), apply + warp (`kapply.py`, `kwarp.py`) |
| `o_decoder/` | start-up decoder of `dec_e4s_c1nx`: estimator tables, source weights and targets (`dw.py`), support gate (`dsup.py`), synthetic-MAST weights (`dsyn.py ws`), the decoder (`dec_core.py`, driver `dec_run.py`) |
| `ch2d_scalars/` | β_N from the Thomson pressure mapped on the solved flux |

`src/stage2_run/score_stage2.py` wraps the organisers' scorer for local evaluation against labelled parquet files. Every call in
this package is `--machine d3d` (DIII-D labels). The file also has a `--machine mast` branch, which loads the scorer's
`mast_envelope.npz` and scores against the labelled files it is given; it remains from August / September 2026, when it was run
against the labels of the three MAST demo shots (`PROVENANCE.md` section 2.1), and no entrypoint of this package calls it.
`src/stage6_run/.../validate_local.py` is a local format check.
