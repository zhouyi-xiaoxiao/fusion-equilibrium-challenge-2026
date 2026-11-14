# MAST pipeline profiles

`ch2/predict_mast.sh` and `ch2/calibrate_d3d.sh` run whichever MAST pipeline the environment variable `PIPE` names
(`ch2/pipelines/<PIPE>.sh`); without `PIPE` the name in `ch2/pipelines/DEFAULT` is used. The file that is submitted is fully
defined by the profile (and the files it sources), the code under `src/`, the constants in `ch2/constants/`, and, for a profile with
learned modules, the model files: shipped ones are listed in the profile (`MODELS`, sha256 in `weights/MANIFEST_CH2.sha256`), models
that the run trains itself on the shots being predicted are listed in `RUN_MODELS`. `predict_mast.sh` writes `record.json` next to the
output with the sha256 of all of them.

| profile | pipeline | Development-phase file (`mast_public_test.npz`) |
|---|---|---|
| `p7a` | kinetic inverse INV_P7a: θ = (edge offset, log pressure scale, two FF′ shape coefficients), FF′ family "fe" | md5 `1915e1d3b5d50a1c7d86e0b4c55a931f` (Development submission #10) |
| `p7c` | INV_P7a with the cubic FF′ family plus an edge term, fixed pressure-gradient shape, cut-cell source quadrature | md5 `b576340dc3ed986be0144f56a91b6732` (Development submission #11) |
| `p7a_tsw05` | `p7a` + temporal smoothing of the plasma flux along each shot (random-walk smoother, 0.5 frame) | md5 `c052f31c99426496da89ff5adfe0a2bb` |
| `p7li` | `p7a` + an l_i datum (DIII-D-trained l_i-from-ramp-history model, one Gaussian row in the fit) + fallback validity gate | md5 `2d61b873052d582998bb1813bf08db48` |
| `syn_xa_p7li50` | 0.5 × network trained on our own synthetic equilibria (`syn_net.inc`, transductive, GPU) + 0.5 × `p7li`, map + scalar blend | md5 `11d94769bb5cc5cb9c0b4b11accd6c87` |
| `syn_xa_p7c80` | 0.8 × the same network + 0.2 × `p7c` | md5 `c46183b816983ea499bafba56cdb5d6b` |
| `int_b50c1nx` | `syn_xa_p7li50` + DIII-D-trained map corrector (`corr_b50.inc`), boundary × 0.5, l_i × 1, no axis term, zero mean | md5 `275f4f59fdecdef368cdff35fda09ad5` (Development submission #15) |
| `int_b50gld` | `syn_xa_p7li50` + the same corrector, every output × 1, zero mean | md5 `af5e3a7e5fd6b12c7daaf737355be1db` (Development submission #16) |
| `dec_e4s_c1nx_a1w5` | the maps of `dec_e4s_c1nx`; q95 by rule A1 (follows the corrector's map change), β_N with a 5-frame median (`scal_a1w5.inc`) | md5 `4d66f8fc0bf014dd329b717bf8695d20` (built in development, not submitted) |
| `dec_e4s_c1nx` | `syn_xa_p7li50` + start-up decoder (`dec_e4s.inc`: the start-up frames re-decoded toward per-functional targets of the network and the inverse) + the corrector of `int_b50c1nx` on the decoded maps | md5 `006ed7289f53e51c9602e8326b75c3c4` (Development submission #17) |

`syn_net.inc` is not a profile: it holds the network stages shared by the two `syn_*` profiles. `corr_b50.inc` is not a profile
either: it holds the post stage `corr` of the profiles `int_b50*` and `dec_e4s_c1nx` (needs `src/.vendor_scorer` from `env/build_env.sh`
and the corrector model of `weights/MANIFEST_CH2.sha256`; its DIII-D derivation is `ch2/train_corr_b50.sh`). `dec_e4s.inc` holds the
post stage `dec` of `dec_e4s_c1nx` (constants `ch2/constants/dec_e4s/`, support source table of `weights/MANIFEST_CH2.sha256`; DIII-D
derivation `ch2/train_dec_e4s.sh` on the training-pool runs of `ch2/train_corr_b50.sh`).

## Profile interface

A profile is a bash file sourced after `env.sh` (which defines `WS` = `src/`, `SCR` = work directory, `FEC_REPO`) and
`ch2/common.sh` (which defines `PY` = FreeGS environment, `PYS` = ML environment, `G1`, `INV`, `INV6`, `R7`, `SCALC`, `GEOM`, `NPROC`).
A profile may source another profile (its base pipeline) and then override or extend it.

| variable / function | required | meaning |
|---|---|---|
| `TAG` | yes | file-name tag of the run artefacts (`parts/<TAG>_mast_<idx>.npz`, constants `calib_<TAG>.json`, `c_<TAG>_mast.json`) |
| `SOLVER_DIR`, `SOLVER` | yes | per-shot solver script and the directory it is run from (it imports its sibling modules). CLI of the `run_inv*` family: `--machine --mode --data-dir --wall --cfg (repeatable) --set key=json (repeatable) --tag --out --procs`, DIII-D selection `--exclude-holdout --stride --frames-per-shot`; output in the "parts" layout. A shard is solved on a directory of links to its shots (`--data-dir`), so no solver needs a shot-list option |
| `SOLVE_ARGS` | yes | array: configuration of the solver, used for MAST and for the DIII-D calibration |
| `MAST_ARGS` | no | array: extra solver arguments for MAST only |
| `PCAL_ARGS` | no | array: extra solver arguments for the DIII-D calibration run (clean MAST-like pseudo-horizontal Thomson chord) |
| `CONST_PIPE` | no | name of the constants directory `ch2/constants/<CONST_PIPE>` (default: the profile's own name); a profile that only adds stages on top of a base pipeline uses the base's constants |
| `EXTRA_CONST` | no | further constants directories (`:`-separated), recorded in `record.json` (e.g. `ch2/constants/syn_xa` of the network) |
| `PACKER` | no | packer with the CLI of `e_mast_inv6/pack_inv6.py` (default) |
| `PSTAGES` | no | array of profile stages `n`; each needs a function `stage_<n> <run dir> <MAST shot dir>`. They run after `solve`, in this order, when `STAGES` names them (default: all). A stage may honour `NSHARD` / `SHARD` (then it can run as independent jobs like `solve`), must skip finished work when re-run, and returns non-zero on failure |
| `POST` | no | array of post-stage names `n`; each needs a function `post_<n> <run dir> <input npz> <output npz> <MAST shot dir>` that writes the output npz (same keys, order and shapes) and returns non-zero on failure. Stages run in order after packing; the first one receives the packed base file |
| `MODELS` | no | array of shipped model files that must exist before the run; listed with sha256 in `weights/MANIFEST_CH2.sha256` and placed by `tools/fetch_weights.sh` |
| `RUN_MODELS` | no | array of model files the run itself produces (paths relative to the run directory); their sha256 goes into `record.json` and they are published with the Final run (RUNBOOK_FINAL.md) |
| `labelfit` | no | function `labelfit <diii_d_train dir> <out dir> <holdout spec>`: regenerates the DIII-D label-fit statistics behind the profile's priors |
| `calib_<n>` | no | further DIII-D stages of `ch2/calibrate_d3d.sh` (e.g. `calib_lifit` of `p7li`: the l_i model) |
| `REF_MD5` | no | md5 of the Development-phase file of this pipeline on the public test (printed for comparison only, and only when the output name is `mast_public_test.npz`) |

Rules every profile follows (rule 3 of the challenge): every process that reads a MAST file installs
`src/stage11_run/ch2_geometry/rule3_guard.py`; no MAST label, demo shot, envelope mask or MAST equilibrium statistic is read; all
fitted constants come from DIII-D labels, public machine description, our own synthetic equilibria, or label-free statistics of our
own MAST solutions and MAST inputs; on the shots being predicted, only their input columns are used (per-frame solutions, and for
the `syn_*` profiles the synthetic equilibria driven by those inputs and the network trained on them — label-free test-time use of the
MAST inputs, which the organisers' permitted-use table allows).

## Running a profile with learned stages (example `syn_xa_p7li50`, 32 nodes)

```bash
# 1. in parallel: solve shards and synthetic-generation shards (CPU), and the network input features (CPU, seconds)
for k in $(seq 0 31); do PIPE=syn_xa_p7li50 NSHARD=32 SHARD=$k STAGES=solve  bash ch2/predict_mast.sh <shots> <run> <order>; done   # one job each
for k in $(seq 0 31); do PIPE=syn_xa_p7li50 NSHARD=32 SHARD=$k STAGES=syngen bash ch2/predict_mast.sh <shots> <run> <order>; done   # one job each
PIPE=syn_xa_p7li50 NSHARD=32 STAGES=synreal bash ch2/predict_mast.sh <shots> <run> <order>
# 2. one GPU node: training (about 3 min) + prediction of every frame
PIPE=syn_xa_p7li50 NSHARD=32 STAGES=syntrain bash ch2/predict_mast.sh <shots> <run> <order>
# 3. finish: merge, beta_N table, pack the base inverse file, blend + pack, validate
PIPE=syn_xa_p7li50 NSHARD=32 STAGES="merge scal pack post validate" bash ch2/predict_mast.sh <shots> <run> <order> <out name>
```

The two DIII-D constants of the network come from its DIII-D twin: `ch2/train_syn_d3d.sh <diii_d_train> <name>` (stages
gen / real / train / calib; prints COMPARE lines against `ch2/constants/syn_xa/calib_Cxa32.json`).

## Adding the finally chosen pipeline (drop-in procedure)

1. **Code.** Add every source file the pipeline runs under `src/`, with the two roots taken from the environment (`WS`, `SCR`; no
   absolute path) and with the module layout of development kept, so that imports resolve as they did there. A generated driver (for
   example a `make_driver.py` output) is shipped as the generated file plus its generator. Modules or packages that the development
   runs took from `PYTHONPATH` (vendored copies) must be added explicitly — as package code, or as a pinned, hash-locked dependency in
   `env/` (example: scikit-image 0.25.2 for the l_i table builder). The clean room has no development tree, so a missing module fails loudly.
2. **Profile.** Write `ch2/pipelines/<name>.sh` from the closest existing profile. Mapping of the module types of the development:
   - *another inverse variant*: a profile like `p7c` / `p7li` (solver, `SOLVE_ARGS`, constants, `labelfit` / `calib_<n>`);
   - *network on own synthetic equilibria + blend*: source the base inverse profile, set `SYN_W`, source `syn_net.inc` (as
     `syn_xa_p7li50`); a different recipe = a copy of `syn_net.inc` with its own generator / training arguments and DIII-D twin;
   - *map-space corrector trained on DIII-D labels* (shipped weights): a `POST` stage after the base pipeline, weights in `MODELS` +
     `weights/MANIFEST_CH2.sha256`, training entrypoint `ch2/train_<module>.sh` (DIII-D runs of the base pipeline on train shots →
     corrector fit); a corpus statistic of the MAST maps (e.g. zero-mean centring) is computed inside the stage from the run's own
     maps; see `TEMPLATE_learned.example`;
   - *test-time ensemble of numerical twins* (K solves of the same inverse with perturbed numerics, member mean, then smoothing):
     `PSTAGES` that solve the K members into `<run dir>/members/<m>` (shardable like `syngen`), a `POST` stage that averages the member
     maps and scalars (`j_synth_net/syn_avg.py` does weighted map / scalar means on the parts layout) and, if used, `post_tsmooth`
     from `p7a_tsw05.sh` last.
3. **Constants.** `PIPE=<name> bash ch2/calibrate_d3d.sh <diii_d_train> cal_<name> pcal` and copy `calib_<TAG>.json`,
   `c_<TAG>_mast.json` to `ch2/constants/<name>/` (replace absolute paths in their `source` / `src` fields; values unchanged).
   If the development file used constants from a different run, the COMPARE lines of a clean-room calibration must be
   IDENTICAL (or within 1 ulp) before the profile is accepted.
4. **Clean-room rehearsal.** Fresh environment from the lock files, weights through `tools/fetch_weights.sh`, then the profile on the
   1,206 public-test shots (shards + profile stages + finish): the md5 must equal the Development-phase file, or the difference must be
   explained (PROVENANCE.md section 3). Record CPU core-hours and GPU-hours (PROVENANCE.md section 4).
5. **Provenance.** Add the profile's constants, priors, module data and any Development-board use to PROVENANCE.md section 2
   and the method to METHODS.md section 3. Set `ch2/pipelines/DEFAULT` to the profile that is submitted.
