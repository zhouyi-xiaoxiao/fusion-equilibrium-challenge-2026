# Data provenance and declared compute

This statement covers every number that enters the two submission files. It is written per machine, as the rules ask.
The submitted configuration is the DIII-D profile `hc10x24` (`ch1/profiles/DEFAULT`) and the MAST profile `dec_e4s_c1nx`
(`ch2/pipelines/DEFAULT`). Sections 1 and 2.3 also document the other profiles this package can produce; each is marked "not
submitted" (not used in the submitted Final-phase files; several were Development-board candidates, section 2.7), and where its
constants are a component of the submitted configuration, that is said.
The third Final submission (F3) pairs the DIII-D profile `xhc` (section 1, "Profile xhc") with the MAST file of the first;
its DIII-D file is computed from package commit `453d04c` (branch `f3-x`; the files that the `hc10x24` and `dec_e4s_c1nx` runs
execute are byte-identical to `82e50ff`).
The second Final submission (F2, Codabench 1003753) pairs the DIII-D file of the first (byte-identical, md5 `33161a88…`) with a
MAST file that is the MAST file of the first (md5 `07e2fed0…`) after one additional post stage, `zt05` (profile
`dec_e4s_c1nx_zt05`): md5 `b3315dad75b0759d9d2e6aaa6d8cc162`, sha256
`de7a84b7cba59906618c1d74f4a9cfb9b5fb0d97e9d68774d9fd51da33c27e52`, 1,058,816,873 bytes. The stage is in package commit `262eb1c`
(branch `f2-zt05`), which is not an ancestor of this commit; its five files are not in this tree. It was applied to the finished
F1 file with `ch2/apply_zt05.sh` (Slurm job 7554076, 19 October 2026, 72 CPU cores, no GPU). `zt05` is the variant of Development
submission #22 (section 2.7), with the same constants (β 0.5, clip 0.04, σ 0.12 m). Against the F1 MAST file: q95 and β_N arrays
bit-identical (2,410 of 2,410); ψ changed in 68,258 of 68,373 frames; the largest change of the last closed flux surface is
2.31e-5 m (0.023 mm; Hausdorff distance over all 68,258 changed frames, measured with the first-wall mask of this package). The
build check compares the maximum on a 1-in-25 sample with 1.5e-5 m, a bound taken from the same sample of the Development #22 file
(maximum 1.48e-5 m), and records that item as not passed (job 7554081). Over all changed frames the Development #22 file reaches
6.44e-5 m (jobs 7554203 and 7554204). The shipped-weights check
of RUNBOOK_FINAL.md B.7 was not run separately for this file; its input is the F1 file, whose check is described in section 3.
Paths refer to this repository. The Final-phase files are computed from package commit `82e50ff`. Later commits add the training
entrypoint `ch2/train_mast_net.sh` of the Final-input network (it only calls the unchanged `ch2/predict_mast.sh`) and documentation
(this statement, README.md, METHODS.md, RUNBOOK_FINAL.md), change the LICENSE copyright line and add the line of each published
Final network to `weights/MANIFEST_CH2.sha256`; no file that the Final run executes or records in its `record.json` changed.

## 1. DIII-D file (Challenge 1)

**Inputs used at inference.** Only input columns of the challenge parquet files: PF / E-coil currents, plasma current, toroidal
field current, Thomson profiles and their time bases, and the query columns `efit_times`, `efit_grid_R`, `efit_grid_Z`.
No `efit_*` label column and no `magnetics_dsep` column is an input (the only feature whose name contains "efit" is derived from
`efit_times`). The organisers' `data_fixes.py` is applied unchanged. On top of it, one inputs-only rule corrects the I_p time
axis on the shots where `fix_d3d_ip_times` keeps the shipped axis (forum post 2831 asks for no other time-axis correction; we
disclosed this rule to the organisers by e-mail on 18 October 2026): if the magnetics record starts before -6 s (the older
data-acquisition era) and placing the start of the I_p record 8,192 ms after the start of the magnetics record puts the
breakdown time (10 % I_p crossing) inside [-100, +150] ms, that axis is used; otherwise, if the breakdown time on the shipped
axis is later than 40 ms, the axis is shifted so that it falls at 15 ms; otherwise the shipped axis is kept. Shots that
`fix_d3d_ip_times` re-origins are not changed. No label is used. Code: `src/stage11_run/e_ip_axis/ip_axis.py`
(`fix_d3d_ip_times_v22`; `install()` re-binds `data_fixes.fix_d3d_ip_times` at run time, so the feature code builds the
FEATURE-V2.2 rows with it). The base models and the 24 heads of the submitted profile `hc10x24` read these FEATURE-V2.2 rows, in
training (docs/CH1_GENERATED_DATA.md) and at inference; the out-of-fold teacher statistics and the 33 frozen blend constants of
the correction step were derived in development from models on FEATURE-V2.1 rows, which apply `fix_d3d_ip_times` only.

**Labels used for training.** The labels (`efit_psirz`, `efit_q95`, `efit_beta_n`) of the 7,041 shots of the public
`diii_d_train` configuration, dataset `Sophelio/fusion-equilibrium-challenge` (v1.1.0, CC BY 4.0), and scalars derived from
those label maps with the organisers' scorer code. Nothing else.

| item | fitted on | where |
|---|---|---|
| 38 shipped weight files (11 frame MLPs, 4 temporal-window models, 17 scalar models — these 32 base files are used by both profiles — and 6 direct heads of profile `fccx`, not submitted) | all 7,041 labelled shots, fixed epochs and seeds, no validation set | `ch1/train.sh`; sha256 in `weights/MANIFEST.sha256` |
| 24 direct heads of profile `hc10x24`, submitted (4 groups × 6 seeds, residual MLP 1024 × 8, dropout 0.15, 42 epochs) | the same 7,041 shots, fixed epochs and seeds, no validation set; inputs = the mirror-canonical FEATURE-V2.2 rows the base models already read | `CH1_PROFILE=hc10x24 ch1/train.sh`; sha256 in `weights/MANIFEST_CH1_HC10X24.sha256` |
| width / depth / dropout of the `hc10x24` heads (1024, 8, 0.15) | a 5 % shot-disjoint validation slice of the training split of one cross-validation fold (a screen of about 15 settings in two passes, 3 seeds each; criterion: validation error of the 7 scored scalars); the slice can contain shots of the second fold (see "Model selection") | fixed in `ch1/profiles/hc10x24.sh` / `ch1/train.sh` |
| feature scalers, PCA bases (80 / 120 components), metric matrices `M_80` / `L_80` / `L_120` | the 6,891 shots that are not in the 150-shot development holdout (`splits/d3d_holdout150.txt`) | `weights/MANIFEST.sha256` (used at inference) and `weights/GENERATED.sha256` (training only); `docs/CH1_GENERATED_DATA.md` |
| out-of-fold teacher statistics (`cov_oof`, finite-difference steps, innovation scales) | 4-fold out-of-fold predictions over the 7,041 shots | `OOF=1 bash ch1/train.sh` |
| 33 frozen blend constants (`blend_frozen_gv21c.json`) | a 344-shot (5 %) shot-disjoint validation slice of the 6,891-shot development run | `weights/MANIFEST.sha256` |
| 11 variances in `ch1/frozen/y_aux_var.json` | scorer-derived scalars of the label maps, all 1,559,340 frames of the 7,041 shots | weights of one ridge term |
| profile `xhc`: 11 frame MLPs and 4 temporal-window models after the contour fine-tune (the 17 scalar models and the 24 heads are those of `hc10x24`) | warm start from the `hc10x24` base weights; the same 7,041 shots, fixed epochs and seeds (except the λ calibration pass of the temporal-model fine-tune, which is not seeded: section 3, row "DIII-D weights, `xhc`"); the added loss reads the reference LCFS of each training label (scorer code on the label map; reference-contour cache and label-polygon bank built from the `diii_d_train` labels by `ch1d_liloss/li_prep.py` and `j_dlcfs_contour/jc_bank_label.py`), its per-shot weights and the 256 held-out gate rows come from the base members' own outputs on the training rows (`jc_c0.py`, `jc_bank.py`) | `ch1/train_xhc.sh` (`ch1/train.sh`, then `ch1/train_xhc_ft.sh`); sha256 in `weights/MANIFEST_CH1_XHC.sha256` |

**Not used for DIII-D:** MAST data of any kind; any external dataset; the public-test or Final input shots for anything but
inference (no test-time adaptation, no pseudo-labels).

**Model selection.** Architecture, loss and post-processing choices were made on the 150-shot holdout and on two 500-shot
cross-validation folds of the labelled shots (one random, one blocked in time), and with the Development leaderboard: the DIII-D
scores of our Development submissions were read and used to choose between complete candidate pipelines (section 2.7 lists
them). No leaderboard score entered a fitted constant.

The two DIII-D profiles of this package differ only in the corrector heads: `hc10x24` is submitted, `fccx` is not submitted.
`hc10x24` was accepted over `fccx` by a rule written
down before its evaluation: paired shot bootstrap (5,000 resamples) on the random fold, ΔS = +0.000054 ± 0.000012 with the lower
bound above zero after counting all 10 candidates scored on that fold, and no loss beyond pre-set limits on the blocked fold
(+0.00027 ± 0.00005) and on the holdout (+0.000005 ± 0.000024, i.e. neutral). Its hyper-parameter screen used the training split
of the random fold; the blocked fold is a subset of that training split, so the blocked-fold check is not fully independent of the
screen (the random fold and the holdout are). `hc10x24` was then kept over `fccx` with their Development scores (section 2.7,
submission 14 against 10–13).

**Profile xhc.** Its only change against `hc10x24` is the contour fine-tune of the base members (METHODS.md section 2). The
recipe was developed on the earlier base (shared by `fccx` and `hc10x24`): the binding version of the loss had no effect; for
the label-polygon loss the weight (share 5, cap 20) and the schedule (40 epochs, learning rate 2e-4) were chosen on the 150-shot
holdout before any fold score of that setting existed (an 8-epoch version had been scored on the folds before, hence six
settings counted on the random fold), then confirmed on both folds and replicated with three fresh fine-tune seeds (seed
standard deviation of the whole chain 1–13e-6). The FULL weights were trained
once with those settings (fine-tune replicate 0) on 12 October and are the shipped ones. On 19 October a paired test against
`hc10x24` was written down before any number of the combination existed (random fold ΔS ≥ +0.00004 with the lower bound for six
looks above zero; blocked fold ≥ 0; holdout pooled ≥ 0, halves ≥ −0.0001; leave-one-block-out ≥ −0.0001; ΔR²ψ ≥ −1e-5;
ΔR²_l_i ≥ −5e-4 on both folds; no new extraction or derivation failure) and evaluated once: random fold +0.000092 ± 0.000009
(lower bound +0.000070), blocked fold +0.000074 ± 0.000028, holdout +0.000069 ± 0.000027 (halves +0.000036 / +0.000103); ΔR²_l_i
+0.00018 / −0.00038 on the folds and −0.00092 on the holdout (reported, not a veto). All three labelled evaluation sets had been
seen for the fine-tuned base during its development, so the gain carries a selection effect of the order of 1–2e-5; no untouched
DIII-D set remains. `xhc` never ran on the Development board; no Development score entered its choice.

## 2. MAST file (Challenge 2, zero-shot)

### 2.1 What was not used

**No MAST or MAST-U equilibrium data of any source was used to train, calibrate or select any model, constant or setting of this
package or of a submitted file**: not the
FAIR-MAST equilibrium reconstructions or a mirror of them, not weights derived from them, not the three labelled demo shots
(by our records no model or constant of this package was fitted, calibrated or selected on them; the code structure of the MAST
solver descends from September pipelines that were diagnosed and chosen between on demo-label scores, see "Early MAST pipelines"
below), not published equilibrium-derived statistics. No entrypoint of this
repository opens `mast_envelope.npz`. Every process that reads a MAST file installs
`src/stage11_run/ch2_geometry/rule3_guard.py` (input-column allow-list; refuses label, demo and envelope paths).

**Test-shot identities.** On 2 September 2026, before the organisers' rulings of 23 September and 4 October, we ran an input-only
matching of the 1,206 public-test and 3 demo MAST input files against FAIR-MAST level-1 `amc` current traces and catalogue
metadata (Slurm job 6247099, 06:12–06:22 UTC; a login-node smoke run on 7 files, 4 public-test and 3 demo, preceded it at
06:00 UTC). The aim was an exclusion list, to keep test shots out of the training set of a branch trained on archive EFIT
reconstructions that was planned at the time; that branch was not built (its conversion job never ran). The
pre-filter compared plasma current and toroidal-field current with the 14,581 archive shots that have an `efm/psirz` record
(`amc` present on 14,580); the
verification compared all 15 `amc` current traces for the 2,622 shortlisted shots. Result: 1,205 public-test files and the 3 demos
matched exactly; 1 public-test file was not matched. On 8 September job 6400520 (12:21–12:33 UTC) ran version 3 of the script: it
fingerprinted all 1,209 challenge input files again, fetched `amc` time, plasma current and toroidal-field current of all 14,581
archive shots again (131,221 objects), and ran the match on 21 public-test files (20 of those matched on 2 September and the
unmatched one) and the 3 demos. Result: 12 probable (9 public-test files and the 3 demos, each to its shot of 2 September),
11 invalid, 1 unmatched; the script's gate for archive training stayed closed (policy status BLOCKED; its conditions include the
full file count). A separate evidence script in the same job compared the 20 public-test pairs and
the 3 demos with the archive `amc` arrays sample by sample (1,128 objects of 24 archive shots): time grids (archive seconds converted to
milliseconds in float32) and current values bit-identical on all 23 pairs (the evidence script was also run once on a login node,
with identical result). No `efm` object was requested in that job. The table
(file name to archive shot number) is not part of this repository; the organisers hold a copy. The same September probes read
the archive EFIT values of the 3 public demo shots and converted two further archive shots, including their EFIT values, for a
format test. The converter probe also read level-2 coil geometry (`pf_active`; no equilibrium data) of three archive shots. A
lookup of 10 November 2026 in the matching table (exact match on the column `archive_shot_id`) shows that none of the shots
converted or read by these probes is among the archive shot numbers matched to the 1,205 matched public-test files or the
3 demos, or among the two closest archive candidates listed for the one unmatched public-test file. The Final inputs were never
matched, so the lookup does not cover Final shots. The converted
files were not used for training, and no later code or job that we found reads them. Apart from the geometry guard described
next, we found no file, constant, model or selection step in this repository or in the pipeline of any submitted file that uses
that list or an archive shot number of a test shot (basis: a search of the code trees of the submitted pipelines and of this
package and a scan of their text files for the matched numbers on 14 October 2026, covering the Development uploads #1–#20 and the
package of that day; a scan of the file-open logs kept from our compliance re-runs of those pipelines for the paths of the matching
outputs, with no hit; the Slurm accounting of 1–12 September, read on 14 October, which shows the two matching jobs; and a keyword
search of the trees of `82e50ff`, `262eb1c`, `453d04c` and `f4eb93f` on 10–11 November 2026, whose hits are the wording in
`ch2/geometry/mast_limiter.json` named below and one unrelated pip hash string in each of the two ML lock files. These are searches
of code, files, job accounting and re-run logs; the original production runs were not traced). The one place where the matching left a trace is our geometry script: the version that fetched the
shipped first-wall file `ch2/geometry/mast_limiter.json` (on 11 October 2026, from shots 22000, 29500 and 15000; section 2.3)
refused shots inside the archive id range of the public-test population, a range known from that matching; the shipped file's
provenance note still names that guard. In this repository the guard is replaced by a fixed list of reference shots
(`src/stage11_run/ch2_geometry/fetch_geometry.py`). The wall geometry is the same for every shot (EFIT machine description
`lim_mast.010_M4`, valid for shots 0–40000), so the guard changed no value in any file. Final-phase inputs are never matched.

**Early MAST pipelines, 31 August – 8 September 2026.** In that period, before the organisers' rulings of 23 September and
4 October, our MAST pipelines (development stages 2–10) were diagnosed and chosen between on the labels of the three MAST demo
shots 28348, 28350 and 28351 (`src/stage2_run/score_stage2.py --machine mast` against the demo files of the starter repository).
None of them was submitted. On 11 October 2026, after reading the rule-3 update, we retired them and rebuilt the
MAST pipeline: its first solver was derived from the September free-boundary solver file with the envelope and demo paths
removed, and the closure constants of this package (section 2.3) come from scans on DIII-D train shots. The MAST branch of
`score_stage2.py` is still in the shipped file; no entrypoint of this repository calls it.

During development, our quality-assurance scripts (not part of this repository) ran the organisers' own scorer derivation, which
loads the envelope mask internally, on our finished MAST files in a separate process, to count frames on which the derivation
fails; the organisers' ruling of 5 October allows this. Its output was a pass / fail count; it did not fit a constant or choose
between candidates.

### 2.2 MAST inputs used

The challenge input columns of each shot: coil and solenoid currents, plasma current, TF current, Thomson `T_e` / `n_e` profiles
with positions and times, the coil-element geometry columns `coil_*` shipped with every shot, and the query columns `efit_times`,
`efit_grid_R`, `efit_grid_Z`.

For the inverse profiles (`p7a`, `p7c`, `p7a_tsw05`, `p7li`; none is submitted on its own, `p7li` is the inverse inside the
submitted `dec_e4s_c1nx`) the file is the per-shot solution of the inverse problem; no parameter is fitted across MAST shots or
across the shots being predicted.

**The `syn_*` profiles are transductive in the test INPUT columns, and say so here:** their network is trained on synthetic
equilibria of our own forward model that are driven by the input columns of the very shots being predicted (measured coil
currents, plasma current, the frames' real Thomson channel positions and each frame's measured profile shape; section 2.4).
No label of any MAST shot exists in that loop: inputs → our forward model → synthetic (input, equilibrium) pairs → network →
prediction for the same inputs. The network of a Final file is trained on the Final inputs by the network steps of
`ch2/predict_mast.sh` (`STAGES=init`, `syngen`, `synreal`, `syntrain`). In the Final run these calls ran as jobs of
`tools/final_slurm.sh` from commit `82e50ff` (RUNBOOK_FINAL.md B.3 / B.4); `ch2/train_mast_net.sh`, added after that commit, is the
training entrypoint that makes exactly these calls, so a rebuild from the training entrypoint repeats this step. Its GPU time is
part of the declared compute (section 4), and the shipped network of each Final submission is the one that the Final run trained
(published with it; sha256 in section 3). The organisers'
permitted-use table allows label-free test-time use of the MAST inputs, and their ruling of 18 October 2026 (forum post 2872)
allows this network on the Final inputs under these conditions; the choice is disclosed because it differs from a model
frozen before the test inputs are seen.

**The `int_*` profiles add two label-free statistics ACROSS the shots being predicted** (section 2.3): the corrector's features are
centred on the medians of our own blend maps over all those shots, and the mean of the predicted corrections over all those shots
is removed. Both are computed from our own maps of the test inputs (no label), and both are recomputed on the Final inputs in a
Final run; they make each frame's correction depend weakly on the other shots of the same set.

**The `dec_e4s_c1nx` profile adds two more label-free statistics ACROSS the shots being predicted** (section 2.3): the decoder
normalises its per-functional terms by the standard deviation of its own targets over all frames being predicted, and its flux term by
the pooled variance of the blend maps over all those frames. The corrector that follows computes its centring and zero mean on the
decoded maps, so re-decoding the start-up frames also moves the other frames slightly (public test: about 0.3 cm at the plasma tips,
l_i × 1.009). All of these are recomputed on the Final inputs in a Final run (RUNBOOK_FINAL.md section B.8 lists them).

Label-free uses of MAST inputs during development, for completeness:
- the plasma-response gain of the closure (5.6 on MAST, 4.3 on DIII-D) is a finite difference of our own converged forward solves;
- design choices of the inverse (which Thomson isotherm levels to fit, how to treat the inner edge, the profile layer of `p7c`)
  were judged on MAST with label-free diagnostics of our own solutions against the Thomson inputs (held-out isotherm levels, edge
  positions, solver convergence and topology);
- the bandwidth of the temporal smoother (`p7a_tsw05`) was checked with a leave-one-frame-out cross-validation of the smoother on
  our own MAST solutions and with held-out isotherms (section 2.3);
- for `p7li`: the l_i-datum variants were checked on MAST label-free before the profile was fixed — the share of MAST frames whose
  input features lie inside the DIII-D support of the l_i model (98.6 %), held-out Thomson isotherms on a 128-shot subset, and
  topology statistics of our own solutions (fraction of limited or open solutions, tail of high elongation). One start-up variant
  (a wall-gap inequality) was rejected on these MAST statistics; the settings that were kept were chosen on DIII-D labels
  (section 2.3);
- for the `syn_*` network: the feature-dropout rates of its training (0.2 / 0.2 / 0.1 / 0.08) follow the missingness of the real
  MAST input features (a comparison of synthetic and real INPUT features only), and candidate recipes were checked on MAST with
  held-out Thomson isotherms and with scalars of our own maps computed with the public wall polygon;
- for the `syn_*` profiles, the P2 feed wiring class of a shot is decided from the shot's own input channels (which of two public turn-count models reproduces the two measured P2 totals from the two feed currents).

### 2.3 Constants of each MAST pipeline (DIII-D labels, public machine description, label-free statistics)

The DIII-D shots used to fit any constant are public `diii_d_train` shots outside the 150-shot development holdout; the
holdout was used only to evaluate candidate settings (no fit uses it). "pcal" = 300 train shots (every 23rd, holdout excluded) × 8 frames, solved by the pipeline itself with DIII-D's Thomson
data re-sampled on a MAST-like horizontal midplane chord, `PIPE=<p> ch2/calibrate_d3d.sh <diii_d_train> <name> pcal`.

**Common to all pipelines** (`src/stage11_run/ch2_G1/configs/cfg_soft_k50.json`, `g_restrict/configs/inv7_base.json`):

| constant | value | how it was obtained |
|---|---|---|
| closure constants `k_paxis`, `alpha_m`, `alpha_n`, `z_eta` | 2.0, 1.38, 1.08, 0.067 | grid scan on 150 DIII-D train shots × 8 frames, maximum of the official composite (ties within one bootstrap sd broken by R_axis R²) |
| Thomson edge rule | density foot at 0.3 × peak; DIII-D edge bias +1.1 cm applied on DIII-D only, **0 on MAST** | DIII-D train shots |
| `sigma_s` 3 cm, `sigma_b` 0.03, Huber 1.5 σ, isotherm levels 0.2 … 0.8 | config | prior choices, checked on DIII-D train scans |

**`p7a`** (INV_P7a, θ = [dr, lcp, b1, b2], FF′ family "fe"; **not submitted** — its two prior rows below are also the priors of
`p7li`, the inverse inside the submitted `dec_e4s_c1nx`, there with the b1 prior relaxed where the l_i datum is active):

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| `C_q95` (factor on the contour-integral q95) | 1.057870220684001 | median of label q95 / own q95 on 2,361 converged pcal frames | `PIPE=p7a … pcal` (clean room: identical) |
| `c_betaN` (factor on the electron β_N,e) | 2.207065836456903 | least squares through the origin of label β_N on β_N,e, 1,713 core-covered pcal frames | `PIPE=p7a … pcal` (clean room: 1 ulp) |
| prior of the pressure scale `lcp0` = ln 2.88, width 0.35 | config | median of a fit of the profile family to the label flux maps, 87 train shots × 8 frames | `PIPE=p7a … labelfit` |
| prior of the FF′ shape `b0` = (−2.10, 0.27), widths (1.5, 1.0) | config | medians and IQR / 1.35 of the same fit (rounded) | `PIPE=p7a … labelfit` |

**`p7c`** (INV_P7c: cubic FF′ family with an edge term, θ = [dr, lcp, b1, b2, be]; `g_profile/configs/p7c_final.json`; **not
submitted** — its profile family and label-fit priors drive the synthetic generator of the submitted network, see `syn_xa_p7li50`
below):

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| `C_q95` | 1.0439342119374946 | as for p7a, on this pipeline's own pcal run (2,361 frames) | `PIPE=p7c … pcal` |
| `c_betaN` | 2.2682178308054994 | as for p7a, 1,722 core-covered pcal frames | `PIPE=p7c … pcal` |
| `lcp0` = 0.941686503385799 (= ln 2.5643), `b0` = (−0.7551429870689236, 0.5021454910897283), `be0` = 0.042556332697644667 | config | medians over 1,339 converged frames of a label fit of the cubic family (wide priors, s = 0.5) on 172 train shots × 8 frames (every 40th train shot; holdout and the 60 shots of `splits/d3d_lf_disjoint60.txt` excluded) | `PIPE=p7c … labelfit` |
| prior widths σ_b = (0.5, 0.5), σ_e = 0.15 | config | values proposed in an AI-assisted review of the method (section 5); half / base / double compared on DIII-D train pilots, base kept | — |
| pressure-gradient shape s = 0.5 | config | DIII-D median of the per-frame Thomson shape fits in the free-s label fits (1,325–1,338 frames of the 172 train shots) | — |
| bounds b ± 3, be ∈ [−0.5, 1.5]; cut-cell quadrature 4 × 8; `edge_w` 0 | config | a priori / numerical choices (widths 0 / 0.01 / 0.02 compared on DIII-D, 0 kept) | — |

**`p7a_tsw05`** (`p7a` + temporal smoothing; **not submitted**): bandwidth σ = 0.5 frame of a random-walk (local-level Kalman / RTS) smoother of the
plasma part of the flux, fallback / invalid / filled frames treated as missing observations (`ch2/pipelines/tsmooth_w05.json`).
Chosen by evaluation on DIII-D labels (σ = 0.5 positive with a confidence interval above zero on every DIII-D evaluation set,
including the 150-shot holdout, and at every frame sub-sampling from 20 to 160 ms) and checked label-free on MAST (median
per-shot leave-one-out optimum 0.6 frame of the smoother on our own solutions; held-out Thomson isotherms not worse). A fixed constant: nothing is fitted on the shots being predicted.

**`p7li`** (INV_P7li: `p7a` + an l_i datum + a fallback validity gate; `ch2/constants/p7li/`; not submitted on its own — it is the
inverse component of the submitted `dec_e4s_c1nx`):

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| `C_q95` | 1.0312427516896954 | as for p7a, on this pipeline's own pcal run (2,352 frames) | `PIPE=p7li … pcal` (clean room: identical) |
| `c_betaN` | 2.292869230881852 | as for p7a, 1,716 core-covered pcal frames | `PIPE=p7li … pcal` (clean room: 1 ulp) |
| l_i model `li_model_v3.json`: quadratic ridge (linear terms and all pairwise products; 77 coefficients for the 11-feature model, 54 for the 9-feature model used when the two Thomson exponents are unavailable; λ = 1e-4), feature means / sds / ranges, cross-validated residual sd 0.177 (0.213 without the Thomson exponents), support metric, target range | file | **DIII-D TRAIN labels only**: 2,637 train shots (`splits/d3d_li_train.txt`) × 16 frames × {clean, stressed} MAST-like Thomson chord = 83,612 rows; target = cylinder-equivalent internal inductance l_i,c of the label flux map; features = dimensionless plasma-current ramp history in resistive-time units + two Thomson profile exponents; shot-grouped 5-fold cross-validation; 330 further train shots (`splits/d3d_li_check.txt`) as a check set. Both lists are outside the 150-shot holdout | `PIPE=p7li … lifit` (clean room: identical once the shot order of the development table is restored, `splits/d3d_li_*_roworder.txt`; with another row order the coefficients differ by ≤ 1.1e-9 relative) |
| weight of the datum: σ = 0.5 × the model's cross-validated sd (inflated outside the DIII-D feature support); variance of the static prior on b1 × 9 where the datum is active | `li_sig_scale` 0.5, `li_b1_var` 9 | chosen on a DIII-D pilot of 60 train shots (nine variants), before the final l_i model was fitted | — |
| fallback gate ("F1"): a closed-form fallback solution is shipped only if it passes the inverse's own validity test | `fb_gate` 1 | a correctness fix found by an internal audit of the pipeline; its thresholds are the inverse's existing ones | — |
| resistive-time constants (Spitzer resistivity with Z_eff 2, ln Λ 17, trapped-particle factor, a = 0.55 m, R0 = 0.85 m), memory lengths, clips | code | textbook / public nominal machine values; none fitted | — |

In development the fit script was also given a table of the same input features computed from the MAST INPUT columns and
printed how much of it lies inside the DIII-D support of the model (a label-free transfer report). The model file is written
before that table is opened and does not depend on it; the regenerate command above reads no MAST file.

**`syn_xa_p7li50`, `syn_xa_p7c80`** (map blend of a network trained on our own synthetic equilibria, `ch2/pipelines/syn_net.inc`,
with the `p7li` / `p7c` inverse; `ch2/constants/syn_xa/` + the inverse's constants). `syn_xa_p7li50` is not submitted on its own —
it is the blend inside the submitted `dec_e4s_c1nx`; `syn_xa_p7c80` is **not submitted**:

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| generator: priors of the profile latents | ln c_p ~ N(0.9417, 0.35²), (b1, b2) ~ N((−0.755, 0.502), 0.5²), edge term ~ N(0.0426, 0.15²), bounds as `p7c`, pressure-gradient shape s = 0.5 | the `p7c` label-fit priors (table above: 172 DIII-D train shots); width 0.35 a priori | `PIPE=p7c … labelfit` |
| generator: closure offset ~ N(0, 3 cm) about the frame's measured outer Thomson edge; pressure amplitude = measured × logN(0, 0.30); Thomson-shape jitters 0.15 / 0.25; 20 % broad shape prior; three noise realisations (T_e / n_e 3 / 7 / 10 %, edges 0.5 / 1.5 / 3 cm and 1 / 3 / 5 cm, 5 % inboard calibration); every 2nd frame with I_p ≥ 50 kA and a measured outer edge is an anchor; 10 draws per anchor; seed 3 | code / `syn_net.inc` | set a priori; no fitted constant | — |
| generator: Thomson shape of each anchor | per frame | fit of the frame's own measured T_e / n_e profile in the flux coordinate of our own centre solution (MAST INPUT columns) | — |
| network: residual MLP 512 × 4, 4-member ensemble, 60 epochs; 48 PCA components of (ψ − ψ_vacuum) / I_p, input / output normalisers, support clips | weights (produced in the run, `RUN_MODELS`) | fitted on the synthetic set of the run only (233,579 X-point-bounded samples of 297,024 accepted synthetic equilibria from 1,205 driver shots on the public test. Final run: 230,171 X-point-bounded samples of 293,349 accepted synthetic equilibria from 1,204 driver shots; the Final MAST configuration has 1,205 shots, and one of them, `mast_shot_e5cb60a3d6.parquet`, has 5 anchor frames and no accepted sample in 50 draws, so it contributes no training sample. The network predicts all 1,205 shots, 68,373 frames, none by fallback) | `ch2/train_mast_net.sh` (= stage `syntrain` of `ch2/predict_mast.sh`; clean room through that stage, as a job of `tools/final_slurm.sh`: all 129 arrays bit-identical to the development network) |
| training on X-point-bounded samples only | recipe "XA" | the organisers' data description (every time slice is diverted) + comparison on the DIII-D holdout | — |
| `C_q95`, `C_betaN` of the network heads | 1.029516594318144, 1.2272909799162133 | median of label q95 / head q95 and least squares of label β_N on the head, with the network's DIII-D twin (same recipe, synthetic set driven by 1,686 DIII-D train shots — holdout and the 150 shots of `splits/d3d_syn_genban.txt` excluded, every 4th remaining shot — of which the 1,654 outside `splits/d3d_syn_trainban.txt` are used for training) on the pcal frames (2,367 / 2,333) | `ch2/train_syn_d3d.sh` (clean room: identical; twin network arrays bit-identical) |
| prediction: frames without a measured outer edge filled from neighbours; Gaussian smoothing along the shot, σ = 1 frame, of the plasma part and of the scalars | `syn_net.inc` | σ compared on the DIII-D holdout, all frames (0.6 / 1 / 2 frames: S 0.8921 / 0.8930 / 0.8931 with the clean chord); a label-free cross-validation on our own MAST maps prefers 0.2–0.6 and was not used for this choice | — |
| blend weight of the network map | 0.5 (`syn_xa_p7li50`, the submitted configuration), 0.8 (`syn_xa_p7c80`, not submitted) | chosen on the 150 DIII-D holdout shots, all frames, clean and stressed Thomson chord, and confirmed on a further 150-shot DIII-D set not used before. With `p7li`: scan 0.3 … 0.75, w = 0.5 (S 0.9386 / 0.9220; further set 0.9254 / 0.9050 against 0.8977 / 0.8902 for the inverse alone). With `p7c`: scan 0.5 … 0.95, w = 0.8 = the best weight under the stressed chord (S 0.9245 / 0.8969; further set 0.9080 / 0.8831, +0.053 / +0.034 against the inverse alone). The weights 0 (`p7li`), 0.5 and 1 (network alone) with the `p7li` inverse were each submitted to the Development board as complete candidates (section 2.7); no intermediate weight was fitted to board scores | — |
| scalars of the blend | weighted geometric mean (q95) / weighted mean (β_N) of the two components after each one's own DIII-D constants | definition | — |
| fallback file for a shot without any valid network scalar | the packed inverse file of the same run | never needed on the public test (0 shots) | — |

**`int_b50c1nx`, `int_b50gld`** (the maps of `syn_xa_p7li50` warped by a map corrector trained on DIII-D labels,
`ch2/pipelines/corr_b50.inc`; scalars = those of `syn_xa_p7li50`, unchanged). Neither profile is submitted on its own; the corrector
model with the setting `c1nx` is part of the submitted `dec_e4s_c1nx` (applied there to the decoded maps); the setting `gld` is
**not submitted**:

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| training pool | 4,104 DIII-D train shots (`splits/d3d_corr_pool.txt`) = train minus the 150-shot holdout, minus the 150-shot extra set (`splits/d3d_extra150.txt`), minus the 2,637 training shots of the l_i model (`splits/d3d_li_train.txt`; the l_i datum would be in-sample there) | definition | `ch2/train_corr_b50.sh … lists` |
| training tables | the `syn_xa_p7li50` construction run on 12 evenly spaced frames of every pool shot with DIII-D's Thomson data re-sampled on the MAST-like horizontal chord, clean and stressed (noise 10 %, branch 5 %, timing 10 ms, separatrix 3 cm, edge noise 1 cm): the `p7li` inverse, the DIII-D twin network, the 0.5 / 0.5 blend (48,987 frames per condition); targets = the label map's axis, top / bottom / inboard / outboard points of the last closed flux surface (scorer's extraction with the DIII-D envelope) and l_i, as displacements from OUR map's points in units of the minor radius and as log l_i ratio | **DIII-D TRAIN labels only** | stages `inv_* net_* tab_*` (development: regenerated bit for bit by an independent self-audit, wall-clock columns aside) |
| model `mB50_tr.pkl` (sha256 `150620dd…`, `weights/MANIFEST_CH2.sha256`) | gradient boosting (LightGBM 4.6.0), 9 outputs, 300 trees, 15 leaves, learning rate 0.05, at least 200 rows per leaf, row / column subsampling 0.7, L2 20, targets clipped at ±0.6; 41 dimensionless features (shot medians of OUR map's shape centred on the corpus medians of the same quantities, shot-relative deviations, inverse diagnostics, timing; `k_corrector/kcommon.py`, set `tr`) | fitted on the pool tables; hyper-parameters a priori (taken unchanged from an earlier corrector on another base, not tuned); shot-grouped 5-fold CV R² of the targets 0.40–0.75 | `ch2/train_corr_b50.sh … fit` (refit from identical tables: predictions within 1.1e-16, 3 of 9 boosters text-identical — LightGBM's multi-threaded sums are not bit-reproducible) |
| support gate | weight 1 inside the DIII-D 99 % ellipsoid of the 41 features (robust Mahalanobis d², q99 185.7), 0 beyond twice the 99.9 % radius² (q999 246.9), linear in between | DIII-D pool statistics; rule a priori | with the model |
| feature centring on MAST | medians of OUR OWN blend maps over all MAST frames being predicted (κ, δ, shift, aspect ratio, l_i, …; 12 evenly spaced frames per shot) | label-free statistics of our own maps of the shots being predicted — recomputed on the Final inputs in a Final run | stage `corr` |
| zero mean (`--debias`) | the mean of the predicted corrections over all valid frames being predicted is removed, per output (the machine mean of our maps is not moved by DIII-D's mean correction) | label-free statistic of our own predictions — recomputed on the Final inputs | stage `corr` |
| strength: `int_b50gld` (not submitted) | every output × 1 (axis, four boundary points, l_i) | the strength the DIII-D holdout chooses among zero-mean settings (0.25 … 1): ΔS = +0.0230 [+0.0186, +0.0288] / +0.0304 [+0.0252, +0.0366] clean / stressed on the 150 holdout shots, all frames, paired shot bootstrap 95 %; +0.0300 / +0.0383 on the untouched extra set | — |
| strength: `int_b50c1nx` (the setting of the submitted `dec_e4s_c1nx`) | boundary points × 0.5, l_i × 1, axis × 0 | transfer-guarded setting: on DIII-D +0.0168 [+0.0137, +0.0214] / +0.0217 [+0.0182, +0.0258] (holdout), +0.0203 / +0.0257 (extra set); on MAST the only setting that worsens no label-free signal of our own maps against the Thomson inputs (outer-edge agreement, axis against the Thomson symmetry centre, isotherm mismatch); a setting with DIII-D's MEAN correction left in was best on DIII-D but moved the MAST boundary off the measured Thomson edge and was never used | — |
| warp | normalised Gaussian radial-basis displacement field through the moved control points, widths 0.45 × (a, b), taper to zero 0.5 a outside the control points' box, displacement cap 0.3 a; l_i re-map coefficient clipped to [−0.25, 0.6] | a priori (`k_corrector/kwarp.py`) | — |
| contour extraction on MAST | the scorer's published functionals (scikit-image 0.25.2, the scorer's pinned version) with the PUBLIC first-wall polygon as mask, never the envelope | public geometry | — |

The two settings `gld` and `c1nx` are two values of the corrector's one strength knob; both were submitted to the Development board
as complete candidates (section 2.7). No MAST label, MAST equilibrium product or board number enters the corrector.

**`dec_e4s_c1nx`** (**submitted**; the blend of `syn_xa_p7li50`, its start-up frames re-decoded by a per-functional decoder, then the corrector of
`int_b50c1nx`; `ch2/pipelines/dec_e4s.inc`, constants `ch2/constants/dec_e4s/`, code `src/stage11_run/o_decoder/`; DIII-D derivation
`ch2/train_dec_e4s.sh`):

| constant | value | how it was obtained | regenerate |
|---|---|---|---|
| frame regimes | start-up = the inverse's last closed surface is not X-point-bounded, or I_p < 0.6 of the shot's maximum within the first 35 % of the shot; inner-censored (inner Thomson edge censored); near-double-null (up-down asymmetry of the inverse contour < 0.05); ordinary | a-priori rule on inputs and our own inverse solution, the same on both machines (17.0 % of the public-test MAST frames are start-up) | code (`dw.py`) |
| DIII-D source weights `wD_L.json` (I = inverse map, N = network map) | per functional (the 7 scorer functionals and 6 boundary control coordinates) and regime: convex weight minimising the uncentred second moment of the weighted error; flux weight of the network per regime (start-up 0.66); regimes with fewer than 30 shots pooled | **DIII-D TRAIN labels only**: the pool runs of `ch2/train_corr_b50.sh` (4,104 shots × 12 frames × clean / stressed chord, 96,586 valid frames; holdout, extra set and l_i training shots excluded) | `ch2/train_dec_e4s.sh … tab_* est fit`; from the recorded estimator tables the package's `fit` stage gives the shipped file bit for bit |
| prior widths of the decoder's 12 response coordinates | robust sd of the blend map's control-point errors on the same pool, in units of the minor radius (0.008 … 0.051 a; l_i 0.14 in log) | DIII-D TRAIN labels | in `wD_L.json` ("prior") |
| synthetic-MAST weights `wS_L.json` | per functional and regime: w_S = argmin max over four populations of the normalised squared error of w N + (1 − w) I, regulariser 0.05 toward ½, feasible interval at tolerance 0.02; l_i fixed at ½ a priori (the synthetic worlds cannot represent the inverse's l_i datum) | OUR OWN synthetic equilibria (section 2.4: four populations driven by the input columns of every 3rd MAST public-test shot); no label | `ch2/train_dec_e4s.sh … ws` from the four shipped population tables (`weights/DERIVATION_CH2.sha256`): bit for bit |
| support gate g = 0.5 × support | support 1 below the leave-one-shot-out 95th percentile (2.62) and 0 above the 99.5th (5.93) of the whitened 32-nearest-neighbour distance of 11 dimensionless net-minus-inverse features, linear in between; whitening and thresholds from the DIII-D pool only | DIII-D TRAIN pool features (own maps); rule a priori | support source table `weights/MANIFEST_CH2.sha256` (`train_dec_e4s.sh … support`, bit for bit) |
| decoder `decoder_E4.json` | start = the 0.5 / 0.5 blend map; 12 smooth, current-neutral response coordinates; 3 Gauss–Newton steps; trust region 0.25 prior sd; finite-difference step 0.25; rejection gates (failed contour, control-point jump > 0.3 a / b, axis jump > 0.25, new axis, current oscillation); applied to the start-up regime only | design frozen on 2026-10-13 on a development subset of 300 shots of the DIII-D TRAIN pool (weights refitted without them); the restriction to the start-up regime was chosen after Development feedback (section 2.6) | — |
| target scales s_j, flux variance v_psi | standard deviation of OUR targets / pooled variance of OUR blend maps over all frames being predicted | label-free statistics of the shots being predicted — recomputed on the Final inputs | stage `dec` |
| corrector | `mB50_tr.pkl`, setting `c1nx`, unchanged (on the DIII-D holdout and extra set, clean and stressed, its gain on the decoded base is at least 0.99 × its gain on the plain blend, so it was not refitted); centring and zero mean recomputed on the decoded maps | as for `int_b50c1nx` above | stage `corr` |
| scalars | q95 / β_N of `syn_xa_p7li50`, bit for bit | as above | — |
| scalars of the option `dec_e4s_c1nx_a1w5` (not submitted; Development candidate #19, section 2.7) | A1: q95 = q95 of the blend × H95[corrected map] / H95[decoded map], H95 = contour integral at ψ_N 0.95 of OUR map (the blend's value where a contour is missing or the ratio is outside [0.5, 2]); B7w5: β_N of the blend with a centred 5-frame median instead of 3 | no fitted constant; both rules accepted on DIII-D on this base (holdout and extra set, all frames, clean / stressed: R²(q95) +0.017 / +0.021 and +0.012 / +0.015; R²(β_N) +0.0003 / +0.0026 and +0.0006 / +0.0028, all 95 % lower bounds > 0) | stage `a1w5` (`ch2/pipelines/scal_a1w5.inc`) |

DIII-D evidence of the profile (150 holdout shots and the untouched 150-shot extra set, all frames, MAST-like chord, paired shot
bootstrap 95 %, ΔS against `int_b50c1nx` on the same shots): holdout −0.0003 [−0.0007, +0.0001] clean / −0.0001 [−0.0004, +0.0002]
stressed; extra set +0.0006 / +0.0004 — a tie on DIII-D, where start-up frames are few and short. On synthetic MAST-geometry truth
(the four populations) +0.003 … +0.005 (90 % lower bounds > 0 in all four); label-free on real MAST, the start-up frames' Thomson
isotherm residuals shrink by 0.8 mm at the 0.35 and 0.5 levels and grow by 0.5 mm at the 0.8 level (outer edge unchanged). No MAST label, MAST equilibrium product or board number enters any constant.

**Public machine description used for MAST** (no shot data):

| constant | value | source |
|---|---|---|
| solenoid: effective turns per shipped element set | factor 0.5 (328 ampere-turns per ampere of feed current on 656 elements) | OMAS EFIT signal table for MAST, `omas/machine_mappings/support_files/mast/0/signals.dat` (first data line, "P1 feed is split in two for turns") and `mhdin.dat`, https://github.com/gafusion/omas ; the same multiplier in FreeGSNKE `freegsnke/mastu_tools.py`, https://github.com/FusionComputingLab/freegsnke , and Pentland et al., arXiv:2407.12432 §4.1.1; Kim et al., Nucl. Fusion 62 (2022) 126012 (328 turns). The organisers' forum post of 11 October notes that `sol_current` reads exactly twice the FAIR-MAST level-2 solenoid current |
| toroidal field: N_TF | 24 | same `signals.dat` (`TF1 … 4.800e-6 … (24 turn coil)`); UKAEA `IMAS_MAST_mappings`, `unprocessed/tf_sql_dump.txt` (R·B_vac = 0.0048 T·m per kA), https://github.com/ukaea/IMAS_MAST_mappings |
| first wall (limiter polygon), 37 points | `ch2/geometry/mast_limiter.json` | EFIT machine description `lim_mast.010_M4`, read on 11 October 2026 from the machine-description group `wall/limiter_{r,z}` of the FAIR-MAST level-2 store for shots 22000, 29500, 15000 (geometry arrays only, identical for the three shots; CC BY-SA 4.0; `src/stage11_run/ch2_geometry/fetch_geometry.py`, see section 2.1 on its former shot guard), see `ch2/geometry/NOTICE` |
| coil geometry and turns | per shot | the `coil_*` columns of the challenge inputs (rectangles, turn multiplicity through `coil_input_column`) |
| P2 winding turn counts (inner 12, outer 8) and the two documented feed configurations | used by the `syn_*` profiles only (generator and network inputs: the split of each P2 Rogowski total between the windings follows the shot's wiring class, which is decided per shot from the shipped feed-current channels); not used by the inverse profiles | `mhdin.dat` (FCTURN) and FAIR-MAST level-2 mapping `mappings/level2/mast.yml`, https://github.com/ukaea/fair-mast-ingestion ; channel list of the public FAIR-MAST level-1 `amc` signals |
| nominal R0 = 0.85 m, a = 0.55 m | normalisation radius, prior scales | public machine parameters |
| β_N, q95 conventions (B at the geometric centre, a from the LCFS extent, q at ψ_N = 0.95) | code | public EFIT source `efit/beta_li.F90`, `efit/shapesurf.F90`, https://github.com/gafusion/efit_old_incomplete |
| DIII-D wall (calibration runs only) | `ch2/geometry/d3d_limiter_mask.npz` | FreeGS 0.8.2 `machine.DIIID()` (LGPL-3.0) |

### 2.4 Synthetic data

All synthetic data are free-boundary Grad–Shafranov equilibria computed by this repository's own forward model (FreeGS solver,
the `p7c` profile family, the coil geometry shipped with each shot, the public first wall). They are not distributed: the
generator is seeded per shot, and a full regeneration is bit-identical (checked on all 1,206 public-test shots).

| set | driven by | used for | labels involved |
|---|---|---|---|
| DIII-D twin set (`ch2/train_syn_d3d.sh gen`) | input columns of 1,686 DIII-D TRAIN shots (holdout and two further 150 / 120-shot evaluation lists excluded from generation / training), 20 random anchor frames × 10 latent draws, the MAST-like Thomson chord | training the twin network, whose only product is the two constants `C_q95`, `C_betaN` (calibrated against DIII-D TRAIN labels on the pcal frames) | DIII-D TRAIN labels, in the calibration only |
| MAST transductive set (stage `syngen` of the `syn_*` profiles) | input columns of the MAST shots being predicted (public test in development; private test in the Final) | training the network that predicts those same shots | none |
| decoder populations sA1, sA2, sC1, sC2 (development only; tables in `weights/DERIVATION_CH2.sha256`) | input columns of every 3rd MAST public-test shot (399–402 driver shots, every 6th frame; 2,940 / 2,940 / 3,286 / 3,286 frames); two profile families (the `p7a` and the `p7c` FF′ families) × nominal / stressed Thomson rendering; the inverse and the network re-run on the rendered diagnostics | the synthetic-MAST weights `wS_L.json` of `dec_e4s_c1nx` (fixed artefact; not regenerated in a Final run) and the decoder's in-geometry tests | none |

No synthetic equilibrium is calibrated against a MAST label or a MAST equilibrium product; the only MAST information in the
generator is input columns (coil currents, plasma current, Thomson channel positions and measured profiles).

### 2.5 Other machines' data

- **DIII-D** train labels: as listed in sections 1 and 2.3.
- **NSTX-U.** The public NSTX-U equilibrium set of Wai, Boyer and Kolemen, *Nucl. Fusion*
  62 (2022) 086042 (https://github.com/PlasmaControl/nstxu-nns at commit `f39ad9d9`, file `eqnet/data/datasets/data_pca_017.dat`,
  sha256 `89376600…`, MIT licence; EFIT01 reconstructions of the 2016 campaign, 204 shots, 31,964 time slices, PCA-compressed;
  no Thomson data, no MAST or MAST-U content) was downloaded on 2026-10-12 and analysed as an aspect-ratio check: the ratio of
  the scorer's l_i to the cylinder-equivalent l_i,c, the offset of our DIII-D-trained l_i model on NSTX-U (its ramp-history part, the
  set has no Thomson data: −0.02 on average with an assumed T_e of 0.6 keV, +0.3 in the first fifth of a shot), and the projection of
  NSTX-U FF′ profiles on our profile families. **No constant, weight, prior or blend weight of any profile
  shipped in this repository was fitted on it** (every shipped profile was built before the download or without reading it), and
  no network was trained on it. The analysis was read as supporting evidence when the l_i-datum pipeline was put on the
  Development board.
- No other external dataset was used (the ITPA global confinement database and other public spherical-tokamak material were
  surveyed and not downloaded or used).

### 2.6 Development-leaderboard feedback (allowed by the organisers' ruling of 2 and 5 October; disclosed here)

The organisers allow using the score each submitted candidate receives to choose among the candidates we submitted, including
several settings of one knob, and forbid constant / shifted / rescaled probes and anything fitted to scores. Our use:

- **Selection between complete pipelines**, in this order: G1 forward solve in float16 versus float32 (float32 kept, both machines);
  G1 with a DIII-D-fitted l_i rule and with a wall-clearance constraint versus plain G1 (both rejected); the solenoid factor 1.0
  versus 0.5 (pooled R²ψ 0.63 against 0.95; the factor 0.5 has the physical sources above, and the switch was motivated by our
  Development result); G1 versus the kinetic inverse (a tie); the inverse with multi-start and own-calibration scalars (kept);
  the inverse with and without the shaping-field parameter (removing it was proposed in an AI-assisted review of the method
  together with the per-scalar detail of our submissions; the result is `p7a`); `p7a` versus `p7c` (profile layer; `p7c` became
  the reference inverse); `p7a` / `p7c` versus `p7li` (l_i datum); and three settings of the network blend weight — 0 (`p7li`),
  0.5 (`syn_xa_p7li50`), 1 (the network alone) — each submitted as a complete, separately audited candidate; then the map corrector on that blend in two settings of its strength
  (`int_b50c1nx`, then `int_b50gld`; the order and the condition for submitting the second were written down before the first
  score was read); then the start-up decoder on the corrected blend (`dec_e4s_c1nx`, #17, against #15 = `int_b50c1nx`, identical
  scalars; kept, by a rule written before the score was read) and its all-regime variant on the full-strength corrector (#18,
  against #16); then three variants of `dec_e4s_c1nx`, each against #17 and none adopted for the first Final submission: the
  scalar rules of the option `dec_e4s_c1nx_a1w5` (#19), a third start-up estimator (#20) and a label-free estimate of the
  magnetic-axis height (#22, with an adoption threshold written down before the score was read: +0.0006 in MAST S against a
  threshold of +0.001; it is the post stage of the second Final submission, Codabench 1003753, see the header of this statement);
  #21 re-uploaded #17 unchanged. On DIII-D: `fccx` versus `hc10x24`.
- **Design choices made after per-term Development results were seen** (`dec_e4s_c1nx`): the decision to re-decode only the start-up
  frames was taken after the per-term results of #13 (blend) and #14 (network alone) showed the network's elongation, volume and l_i
  far worse than the inverse's on MAST, together with DIII-D and synthetic evidence that pointed to the start-up regime; it is a
  choice of one of four pre-defined regimes, every weight, threshold and decoder setting predates it. The corrector setting `c1nx`
  was preferred for that profile because #15 scored above #16 (selection between two submitted settings of one knob).
- **Per-shot detail of one of our own results**: the per-shot list of the 20 worst shots and the per-shot mean R²ψ in the
  Detailed Results of our submission of 2026-10-11 19:04 UTC were used in a forensic analysis of the MAST error.
- **No leaderboard number entered a constant**: the solenoid factor is the documented 0.5 (a factor that would have been fitted
  to our board R²ψ was withdrawn and never submitted; a second documented-but-unsupported factor was not submitted); the
  calibration constants of every pipeline are reproduced from DIII-D tables alone (section 2.3).
- Public aggregate leaderboard columns of other teams (composite and component scores) were looked at as context; no constant
  derives from them.

### 2.7 Our Development submissions (Codabench competition 17739)

| # | date (UTC) | DIII-D file | MAST file | used for |
|---|---|---|---|---|
| 1 | 2026-10-11 16:44 | first learned package (PCA-80 MLP bag), float16 | — | DIII-D anchor (holdout-to-board calibration) |
| 2 | 2026-10-11 18:39 | same | G1 soft edge, solenoid factor 1.0, float16 | first MAST entry |
| 3 | 2026-10-11 19:04 | same, float32 | G1 soft edge, float32 | float16 vs float32 (both machines); per-shot detail used in the MAST forensic |
| 4 | 2026-10-11 19:16 | same | G1 + l_i rule + wall clearance | rejected vs #3 |
| 5 | 2026-10-11 19:17 | same | G1 + DIII-D-fitted l_i rule | rejected vs #3 |
| 6 | 2026-10-12 16:35 | blend + consistency correction (robmi) | G1, solenoid factor 0.5 | factor 0.5 confirmed |
| 7 | 2026-10-12 16:36 | same | kinetic inverse INV_P5 | G1 vs inverse (tie) |
| 8 | 2026-10-12 16:54 | mirror-canonical features (E_cfl) | as #6 | DIII-D: chosen on folds before upload |
| 9 | 2026-10-12 17:51 | as #8 | INV_P6 (multi-start, own-calibration q95) | inverse variant kept |
| 10 | 2026-10-12 20:25 | FcCX, file md5 `863586e8…` (the recipe of profile `fccx`; this package's predict path writes md5 `0b13e8ef…`, which differs from the submitted file in 3 of 193,359 frames through the order of the two final guard steps; q95 / β_N identical) | INV_P7a (`p7a`) | shaping-field parameter removed |
| 11 | 2026-10-13 00:01 | as #10 | INV_P7c (`p7c`) | profile layer: `p7c` against `p7a` |
| 12 | 2026-10-13 03:50 | as #10 | INV_P7li (`p7li`) | l_i datum (+ fallback gate) against `p7a` / `p7c` |
| 13 | 2026-10-13 09:58 | as #10 | SYN_XA_P7li50 (`syn_xa_p7li50`) | network blend, weight 0.5 against weight 0 (#12): the blend scored higher and became the reference MAST candidate |
| 14 | 2026-10-13 10:29 | hc10x24 (`hc10x24`, md5 `0c9beef8…`) | SYN_XA (network alone = weight 1; not a profile of this package) | DIII-D: `hc10x24` against `fccx` (#10–#13): `hc10x24` scored slightly higher and was kept. MAST: weight 1 against 0 and 0.5: the network alone scored below both and was dropped; its per-scalar detail (elongation, l_i and q95 of the network far worse than the inverse's) was read and steers later work |
| 15 | 2026-10-13 12:28 | as #14 | INT_B50c1nx (`int_b50c1nx`) | corrector, transfer-guarded setting, against #13 (identical scalars, so the difference is the map correction alone): scored higher (+0.006; read on its detailed-results page — Codabench later marked the submission Failed by a scoring-host error, so it never appeared on the leaderboard); a rule written before the score was read then called for the full-strength setting |
| 16 | 2026-10-14 11:40 | as #14 | INT_B50gld (`int_b50gld`) | corrector at the strength the DIII-D holdout chooses, against #15: scored slightly lower (−0.002, a tie by the rule written before); the corrector strength `c1nx` was kept |
| 17 | 2026-10-14 13:36 | as #14 | DEC_E4s_c1nx (`dec_e4s_c1nx`) | start-up decoder before the corrector, against #15 (identical scalars): scored higher (+0.003, elongation up); a rule written before the score was read kept the start-up decode and called for #18 |
| 18 | 2026-10-14 14:23 | as #14 | DEC_E1s_gld (all-regime decoder + full-strength corrector; not a profile of this package) | against #16 (same corrector setting): slightly lower (−0.002, a tie) and below #17; decoding all regimes was dropped, the start-up-only decode (`dec_e4s_c1nx`) kept |
| 19 | 2026-10-14 15:37 | as #14 | DEC_E4s_c1nx_A1w5 (`dec_e4s_c1nx_a1w5`: the maps of #17, q95 by rule A1, β_N by a 5-frame median) | scalar rules against #17 (identical maps): MAST S 0.7555 against 0.7601 (R² of q95 0.488 against 0.549); #17's scalars kept, the option is not submitted |
| 20 | 2026-10-14 17:02 | as #14 | DEC_Q3s_c1nx (#17 with a network trained on synthetic samples of all topologies as a third estimator of the start-up targets; not a profile of this package) | against #17 (identical scalars): MAST S 0.7578, lower (−0.0023, elongation down); dropped |
| 21 | 2026-10-17 20:52 | as #14 | DEC_E4s_c1nx (`dec_e4s_c1nx`), identical re-upload of #17 (same files) | the submitted pair as the last planned Development upload: DIII-D S 0.9970, MAST S 0.7601, G 0.7623, the same as #17; no choice was made on it |
| 22 | 2026-10-17 22:43 | as #14 | DEC_E4s_c1nx_ZT05 (#17 with one change: the magnetic-axis height estimated from the up–down flux balance of our own map, its constants fixed before any board read; not a profile of this package) | against #17: MAST S 0.7607 (+0.0006 from Consistency, every other component equal), below the adoption threshold (+0.001) written down before the score was read; not adopted for the first Final submission, whose MAST file is that of #17 (as is the third's); used for the second Final submission, Codabench 1003753 (header of this statement) |

The Development phase closed on 18 October 2026 at 00:00 UTC; #22 was our last Development submission.

## 3. Numerical reproducibility

Measured in clean rooms on the cluster (fresh checkout of this package, environments built from `env/*.lock` only, weights placed
through `tools/fetch_weights.sh` and verified by sha256, public-test inputs from the Hub), 12–15 October 2026. "Development file"
= the file that was built in development (and, where section 2.7 lists it, submitted to the Development board).

| item | result |
|---|---|
| DIII-D public test, profile `fccx` (874 shots, 193,359 frames) | md5 `0b13e8ef8df05634a0a3fc4c426455dd` = the reference file of the predict path, in six independent clean-room runs (288 cores / 4 GPUs, 144 cores / 1 GPU, further checkouts, with the stream order derived from the Hub); gate 0 FAIL; organisers' validator OK. All six used the predict script before the selectable-heads switch; with the final script the `fccx` profile has not been run to completion on the 874 shots (the rehearsal job was lost to an operator error on 13 October, RUNBOOK_FINAL.md "known operational faults"). The final script differs from the rehearsed one only in lines guarded by the heads switch, and its shared stages are exercised by the `hc10x24` run below |
| DIII-D public test, profile `hc10x24` | md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c` = the Development file (submitted as #14 and #15), with the final scripts through `CH1_PROFILE=hc10x24 ch1/predict.sh` in a clean room (manifest weights placed by `tools/fetch_weights.sh`; 288 cores + 1 GPU, 2,474 s to the written file + 165 s gate); log lines `heads v2.2 c10 (24 seeds)` and `HEADS_CHECK OK`; gate 8 PASS / 2 REVIEW / 0 FAIL, G5 order PASS, G6 zero derivation failures. The organisers' validator was not run again on this copy: it is byte-identical to the validated, submitted file |
| DIII-D weights, `fccx` | `ch1/train.sh` from the generated data: 34 of 38 files sha256-identical to `weights/MANIFEST.sha256`, the 4 temporal-model checkpoints tensor-identical (38 tensors each, maximum difference 0; serialisation metadata differs); four independent clean-room runs, the last two with nothing but the 41 files of `weights/GENERATED.sha256` and the manifest weights in the work directory |
| DIII-D weights, `hc10x24` (32 base files + 24 heads) | head-training cache rebuilt by `ch1/build_feat22.py` in the clean room: sha256 identical to the development cache. Full retraining from scratch was done once, in development on 13 October with the development copy of the training script (same commands as `CH1_PROFILE=hc10x24 ch1/train.sh` submits): 24 of 24 heads sha256-identical to `weights/MANIFEST_CH1_HC10X24.sha256`, base 28 of 32 sha256-identical and the 4 temporal-model checkpoints tensor-identical; the predict run from the retrained weights gave md5 `0c9beef8…`. It was **not repeated through the package entrypoint in a clean room**: there, the cache build ran, the job submission exposed a malformed job-dependency list in the comparison step (fixed), and the job set of the fixed script was checked with a stub scheduler only |
| DIII-D weights, `xhc` (`ch1/train_xhc.sh`: the 32 files of `weights/MANIFEST_CH1_XHC.sha256` and the 24 heads) | `USE_SLURM=1 CMP=1 bash ch1/train_xhc.sh` on 19 October 2026. Base members and heads as trained by `train.sh`: 56 files, 52 sha256-identical to the shipped ones, the 4 base temporal-model checkpoints tensor-identical (comparison job 7554669). After the fine-tune stage 28 of the 32 files of `weights/MANIFEST_CH1_XHC.sha256` are sha256-identical (11 frame MLPs, 17 scalar models); the 4 fine-tuned temporal models `stage11_ch1/xhc/full/temporal/full/weights/tR_w9d_s{0,1,2,3}.pt` are **not reproduced**: 32 of 38 tensors differ, maximum weight difference 0.025–0.027. Cause (job 7588807): `src/stage11_run/j_dlcfs_contour/jc_ft_tw.py` runs its λ calibration in training mode (window dropout 0.1) before its first `torch.manual_seed` (line 174), so that dropout stream is unseeded; with a seed set before the script two runs are bit-identical. The shipped temporal-model files can therefore not be regenerated bit for bit, and two retrains differ from each other as much as each differs from the shipped files. Measured effect of the retrain: on the 150 holdout shots (in-sample for these members) S 0.998819 → 0.998818 (halves: −3e-6 and 0), a change of 1e-6 against the organisers' retraining tolerance of 0.01 in S; on the public test (874 shots, 193,359 frames), retrained against shipped weights, q95 and β_N bit-identical (874 of 874 shots), flux map per-frame RMS difference / span median 3.9e-5, 99th percentile 5.8e-4, maximum 0.012, per-frame max |Δψ| / span median 1.46e-4, 99th percentile 3.20e-3, maximum 0.052 (the difference between `xhc` and `hc10x24` on the same set: median 1.41e-3, RUNBOOK_FINAL.md B.10). The script was not changed. Inference from the shipped weights is bit-exact (RUNBOOK_FINAL.md B.10) |
| MAST public test, `p7a` (1,206 shots, 69,075 frames) | md5 `1915e1d3b5d50a1c7d86e0b4c55a931f` = the Development file, with 12 × 104 and 32 × 40 cores, four checkouts (the last with the stream order derived from the Hub); organisers' validator OK |
| MAST, `p7c` / `p7a_tsw05` / `p7li` | md5 `b576340dc3ed986be0144f56a91b6732` / `c052f31c99426496da89ff5adfe0a2bb` / `2d61b873052d582998bb1813bf08db48` = their Development files (`p7c` in two clean rooms) |
| MAST, `syn_xa_p7li50`, end to end from parquet (solve, input features, synthetic generation, GPU training, prediction, blend, pack) | md5 `11d94769bb5cc5cb9c0b4b11accd6c87` = the Development file; the network trained in the clean room has all 129 arrays bit-identical to the development network (the files differ only in recorded paths and timing) |
| MAST, `syn_xa_p7c80` | finish stages (merge, scalars, pack, blend 0.8 / 0.2) run from the final tree on the clean-room solve shards of `p7c` and the clean-room network of `syn_xa_p7li50`: md5 `c46183b816983ea499bafba56cdb5d6b` = the Development file. A full run of this profile from parquet in its own run directory has not been done (its solve stage is the `p7c` solve and its network stages are those of `syn_xa_p7li50`, each reproduced separately) |
| MAST, `int_b50c1nx` / `int_b50gld` (corrector stage), 25-shot stride sample | run directories assembled from the clean-room solve, network and inverse outputs of `syn_xa_p7li50`; the profiles' own post stages from the final tree: the re-computed blend has md5 `11d94769…` (= `syn_xa_p7li50`); the corrector table of ALL 69,075 frames (and therefore the feature centring and the zero mean) equals the development table (every numeric column exact, text columns value for value); the gate and correction statistics are identical; the corrector, applied to every 49th shot (25 shots, 1,424 frames), gives corrected maps identical in every array to the Development files `275f4f59…` / `af5e3a7e…`, and the packed members of those 25 shots equal theirs. **Whole files (clean room of 14 October, later the same day): md5 `275f4f59fdecdef368cdff35fda09ad5` and `af5e3a7e5fd6b12c7daaf737355be1db` = the Development files #15 / #16**, from the development `p7li` solve outputs and network prediction (re-used as inputs) with the pack, the blend and the corrector stage run from the package tree (72 cores, 4.5 min per profile). The contour extraction with scikit-image 0.26.0 instead of the pinned 0.25.2 gave the same table on the 25 shots |
| MAST, `dec_e4s_c1nx`, WHOLE FILE | **md5 `006ed7289f53e51c9602e8326b75c3c4` = the Development file (#17)** (clean room of 14 October, 72 cores, 9.7 min). Re-used as inputs (unchanged development outputs): the `p7li` solve outputs, the network's prediction parts and its input features. Run from the package tree: the pack of the `p7li` file (all 1,206 shots equal to the Development `p7li` file), the blend (md5 `11d94769…`), the decoder stage and the corrector stage. Every intermediate equals its development counterpart: the three functional tables (all 69,075 frames), the estimator table, the support gate, the targets and their scales, the decoder's run constants (v_psi, scales, prior widths, configuration), the decoder's per-frame record, the decoded maps (1,206 / 1,206 shots, every array), the corrector table, its gate and correction statistics, the corrected maps. The package's derivation code recomputes `wD_L.json` (from the recorded DIII-D pool estimator tables), `wS_L.json` (from the four population tables) and the support source table bit for bit. Not rehearsed: the DIII-D pool runs and their functional tables (as for the corrector), and the generation of the four synthetic populations (development code; the tables are shipped). The organisers' validator was not run again on this copy: it is byte-identical to the validated, submitted file |
| Final procedure end to end (dress rehearsal, 14–15 October): `hc10x24` + `dec_e4s_c1nx` through `tools/final_slurm.sh` (RUNBOOK_FINAL.md section B.9) | From a fresh checkout and public-test inputs downloaded from the Hub, with every stage computed in the run (stream orders, inverse solve, synthetic set, the network trained afresh, blend, start-up decoder, corrector, corpus statistics): **DIII-D md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c` and MAST md5 `006ed7289f53e51c9602e8326b75c3c4` = the Development files**; the retrained network's weight arrays equal the Development network's; every intermediate equals its development counterpart; both organisers' validators OK; 74 jobs, 2 h 32 min, 65,730 allocated cpu-minutes |
| Frozen package commit `c0e6733` (file tree re-verified on 15 October; the changes after the dress rehearsal touch the local format check, the run driver, the weight fetch, two printed lines of the entrypoints, the geometry fetch script and documentation) | From a fresh checkout: DIII-D md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c` through `ch1/predict.sh` (288 cores + 1 GPU); MAST md5 `006ed7289f53e51c9602e8326b75c3c4` through the finish stages of `ch2/predict_mast.sh` on the Development solve and network outputs (which the dress rehearsal had reproduced from parquet), and the same bytes in sorted order under the private-test output name with a permuted stream order (final file = the exact key permutation); RUNBOOK_FINAL.md section B.9 |
| MAST, `dec_e4s_c1nx_a1w5`, WHOLE FILE | **md5 `4d66f8fc0bf014dd329b717bf8695d20` = the development file** (a second clean room on 14 October, 72 cores, 11 min, every post stage run again: the blend, the decoder and the corrector stages gave the same md5 as in the first; the two record tables equal the development tables in all 68 columns; ψ equals `dec_e4s_c1nx` on every shot). Submitted to the Development board as #19 (section 2.7), not part of the submitted configuration; organisers' validator OK on the development copy (same bytes) |
| MAST constants | pcal `p7a`: C_q95 identical, c_betaN within 1 ulp; `p7c`: both identical; `p7li`: C_q95 identical, c_betaN within 1 ulp; label fit `p7c`: lcp0, b0, be0 identical; l_i model of `p7li`: identical (all fields) with the shot order of the development table restored, otherwise coefficients differ by ≤ 1.1e-9 relative (the table builder writes shots in the completion order of its worker pool); network twin: C_q95, C_betaN and both frame counts identical, twin network arrays bit-identical |
| MAST with a different BLAS kernel (`OPENBLAS_CORETYPE=ARMV8`, the generic ARMv8 kernel, instead of the kernel OpenBLAS selects by default on this CPU; by the probe of 10 November 2026 described below that default is `armv8sve` in the numpy of the solver environment and `neoversev2` in its scipy, and which library each solve call uses was not traced), `p7a` | NOT bit-identical: 3,083 of 69,075 frames identical; per-frame maximum deviation median 1.2 % of the frame's flux span (90th percentile 4.8 %); q95 median 1.2 % (90th 5.5 %), β_N median 0.85 % (90th 3.6 %). The inverse is an iterative non-linear fit with warm starts along each shot and discrete branch decisions, so last-bit differences grow. Effect on the score where labels exist (same pipeline on the 150 DIII-D holdout shots, all 34,055 frames, MAST-like Thomson chord): S 0.876282 → 0.876401 (ΔS = +0.00012; per-scalar Consistency R² changes ≤ 0.008) |
| `syn_*` network on other GPU hardware | not measured. A different seed (same recipe) moves the network map by a per-shot maximum of 1.5 % (median) of the flux span, which bounds what a non-identical retraining can do; predictions from the published network file do not involve training |

A rebuild on other hardware therefore reproduces the DIII-D file to floating-point tolerance and an inverse MAST file to the
statistical level of the BLAS row, not bit for bit. For a `syn_*` file the network of the submission is published
(`weights/MANIFEST_CH2.sha256`), so that the predictions can be regenerated without retraining
(`SYN_MODEL_FILE=<published copy>` in a new run directory, README.md "MAST inference with the shipped network"; the synthetic set is
still generated, because the network stage checks it before it takes the file). Network of the Final file: sha256
`fbcb1ba660a1774b9191a831e30a7622623d626739ae6c0b9ebe57594c21e169` as trained by the Final run (35,874,798 bytes; the run's
`record.json`, `run_models`; Slurm job 7538629) and `a4f54b23b23336dddc6e7123772b3f6bcf2ad3f3bd078775048c688396299b34` for the
published copy `stage11_ch2/final/model_mCxa32.pt` (35,874,670 bytes; `tools/publish_model.py` shortens the recorded path strings;
all 129 arrays bit-identical; job 7552619). A further Final submission with a network of its own would be listed the same way;
there is none: the MAST files of the three Final submissions all derive from this network. The check of RUNBOOK_FINAL.md B.7
(the published copy, through the inference entrypoint on the run's network input features, synthetic set and solve outputs, must
reproduce the md5 of the submitted MAST file) gave that result: md5 `07e2fed0ffac23daebfd2d9433aca99f`, sha256
`8112d560bc406826db9c65e3eca46a143c359852cb7871beaa49604e42c6a8ce`, byte-identical to the submitted file (`cmp`), with
`a4f54b23…` under `run_models` in the `record.json` of the check run. It was run from a clean-room checkout of `82e50ff` with
byte copies of the run's solve shards, network inputs and synthetic set, in two parts: the stage `syntrain` with the published
copy on 19 October 2026 (job 7552621, 19:17–19:19 UTC; 1,205 prediction part files byte-identical to the Final run's), and the
stages `merge scal pack post validate` in that run directory on 10 November 2026 (job 8286806, 72 cores, no GPU, 23:11–23:21 UTC).
On 19 October the closing stages had been run in the parallel run directory that holds a byte copy of the original file
`fbcb1ba6…` (job 7552622, 19:20–19:29 UTC), with the same md5. The first Final submission was uploaded on 19 October at 08:34 UTC,
before these runs; the second (19:38 UTC) and the third (21 October, 01:50 UTC) after the runs of 19 October. The manifest line of the published copy and the values of this paragraph were added on 10–11 November 2026;
the published copy was not in the Hugging Face repository before the revision that carries this commit. Environment of the Final run: Arm Neoverse-V2 cores
with the `aarch64` lock files (device of the network step: section 4); OpenBLAS chooses its kernel for that CPU (the package sets
no kernel override such as `OPENBLAS_CORETYPE`). The kernel was not recorded during the Final run. A probe on 10 November 2026 on
a login node of the same system (login46, Neoverse-V2 CPU; not a compute node of the run), with the two environments of the
Final-run checkout and no `OPENBLAS_*` variable set, reported: ML environment (`src/.venv`; network step and post stages) numpy
2.4.6 with OpenBLAS 0.3.31 `neoversev2`, scipy 1.17.1 with OpenBLAS 0.3.30 `neoversev2`; solver environment (`src/.venv_gs`;
inverse solve and synthetic generation) numpy 1.26.4 with OpenBLAS 0.3.23.dev `armv8sve`, scipy 1.17.1 with OpenBLAS 0.3.30
`neoversev2`. Which of the two libraries each solver call uses was not traced.

Three faults of the rehearsals that are not reproducibility faults of the method, for the record: an exhausted inode quota (lost
shots, re-run), a scheduler-removed temporary directory (one GPU training died; entrypoints now use a private one and fail
loudly), and Hub rate limiting of concurrent validators (HTTP 429; run them one at a time).

## 4. Declared compute

Hardware for everything below: nodes with 4 × NVIDIA GH200 (120 GB) and 288 Arm Neoverse-V2 cores. GPU-hours are allocated
GPU-hours on that hardware (elapsed × GPUs of the job). No conversion to A100 GPU-hours was measured; nothing below is scaled in
our favour.

| part | kind | measured |
|---|---|---|
| Ch1 `train.sh`: 11 frame MLPs + 4 temporal models + 17 scalar models (both profiles) | GPU | 24–25 min × 4 GPUs = 1.57–1.65 GPU-h (four clean-room runs) |
| Ch1 `train.sh`: 6 direct heads (`fccx`) | GPU | 10–11 min × 1 GPU = 0.16–0.18 GPU-h |
| Ch1 `train.sh`: 24 direct heads (`hc10x24`), 4 jobs × 1 GPU | GPU | 1.35 GPU-h (17:41 + 27:44 + 17:50 + 17:50 min, the from-scratch retrain of 13 October in development; the earlier estimate was 1.1) |
| Ch1 `ch1/train_xhc.sh` (profile `xhc`): `train.sh` for the base members and the 24 heads, then the contour fine-tune `ch1/train_xhc_ft.sh` of the 11 frame MLPs and 4 temporal models | GPU | 4.585 GH200 GPU-h for `USE_SLURM=1 CMP=1 bash ch1/train_xhc.sh` on 19 October 2026 (allocated elapsed × GPUs): base 1,401 s × 4 GPUs = 1.557; heads 1.149; fine-tune 1.879 (c0 0.007 + frame MLPs, 4 jobs, 0.723 + temporal models, 4 jobs, 1.149). CPU-only parts of the fine-tune: about 10 min wall on 72 cores. Not measured on an A100. Peak GPU memory was not logged in that run; a separate single-GPU probe with shortened epochs on 20 October 2026 (jobs 7589648–7589650, 7589797) observed 62,112 MiB for the fine-tune of the frame MLPs, the largest stage that `xhc` adds, and 73,506 MiB for the base step in which 4 temporal models and the scalar workers share the GPU; both are below 80e9 bytes (76,294 MiB). A forced worst case of that base step, which is the same for `hc10x24`, peaked at 76,850 MiB: it ran without error with 81,092 MiB free (our figure for the memory an 80 GB A100 exposes to CUDA) and ran out of memory with 76,294 MiB free |
| Ch1 out-of-fold teacher statistics (`OOF=1`) | GPU | 0.20 GPU-h |
| Ch1 development-run heads and teacher behind the 33 blend constants | GPU | about 0.4 GPU-h (estimate) |
| Ch1 metric: 2-fold out-of-fold models | GPU | not separately measured (same model size as one teacher fit) |
| Ch1 selection of the `hc10x24` head hyper-parameters (two screen passes, 3 seeds per setting) | GPU | about 2.5 GPU-h (development; not needed to rebuild the weights) |
| **Ch1 total GPU to rebuild the shipped weights** | | **`fccx` about 2.4 GPU-h, `hc10x24` about 3.6 GPU-h (measured part of the latter: 3.0 GPU-h = base 1.65 + heads 1.35), plus the metric models — inside 24 GPU-h** |
| Ch1 feature caches, PCA | CPU data generation | 7 core-h per feature build |
| Ch1 metric: finite differences and per-frame Jacobians | CPU data generation | 124 + 562 core-h |
| Ch1 inference, 874 shots | CPU + GPU | about 120–150 core-h, GPU busy < 2 min (`hc10x24`: 142 core-h used of 214 allocated, 2,639 s incl. the gate; the 24 heads predict in 4 × 5 s on one GPU) |
| Ch2 calibration `pcal` (300 shots × 8 frames) | CPU only | 10–18 core-h per pipeline |
| Ch2 label fit (`labelfit`) | CPU only | 3–8 core-h (`p7a` 87 shots, `p7c` 172 shots × 8 frames) |
| Ch2 l_i model of `p7li` (`lifit`: 2,967 shots of tables + ridge fit) | CPU only | about 3 core-h (71 s on 144 cores) |
| Ch2 inference, 1,206 shots, inverse | CPU only | 206 core-h (`p7a`), 194–201 (`p7c`), 211 (`p7a_tsw05`), 206 (`p7li`); shards are independent |
| Ch2 `syn_*`: synthetic generation on the shots being predicted (1,206 shots; step `syngen` of `ch2/train_mast_net.sh`; in a Final run on the Final inputs) | CPU data generation (no GPU) | 333 core-h; wall-clock 12 min as 32 parallel jobs of 72 Arm Neoverse-V2 cores (dress rehearsal, longest shard 728 s); Final run (1,205 shots): 12 min 26 s from the first start to the last end, 06:09:45–06:22:11 UTC on 19 October 2026 (32 jobs × 72 cores, job ids within 7538595–7538627; longest shard 746 s, job 7538619) |
| Ch2 `syn_*`: network training (4 members) + prediction, on the inputs being predicted (`ch2/train_mast_net.sh`, stage `syntrain`; in a Final run on the Final inputs) | GPU | 173 s training, 4.2 min job on 1 GPU = 0.07 GPU-h (dress rehearsal on the public test: 162 s training, 258 s job = 0.072 GPU-h; one job cannot exceed 0.17 GPU-h, its 10-minute time limit; a resubmission would add its own) |
| Ch2 `syn_*`: DIII-D twin — generation / training + calibration | CPU data generation / GPU | 224 core-h / 196 s training, 3.7 min job on 1 GPU = 0.06 GPU-h |
| Ch2 `int_*`: corrector training — the `p7li` inverse, the DIII-D twin network and the blend on 4,104 DIII-D train shots × 12 frames × {clean, stressed} chord, corrector tables, LightGBM fit (`ch2/train_corr_b50.sh`) | CPU only (data generation + fit) | 452 allocated core-h in development (inverse 2 × 4 jobs × 144 cores × 12 min = 230; network inputs + prediction 2 × 16 min × 216 cores = 116; blend + tables 2 × 14 min × 216 cores = 98; fit 3.8 min × 144 cores = 9); the twin network is the one of `ch2/train_syn_d3d.sh` |
| Ch2 `int_*`: corrector stage at inference (table of every frame, apply, pack) | CPU only | at most 457 s on 72 cores (development build job, which also ran extra diagnostics) |
| Ch2 `dec_e4s_c1nx`: decoder derivation — functional and estimator tables of the corrector's pool runs, weight fit, support table; synthetic populations for `wS_L.json`; development of the decoder design on a 300-shot DIII-D subset and its evaluation | CPU only | weight fit, support table and `wS_L.json` from the tables: < 1 core-min each; the population generation and the decoder development runs: not separately accounted (development) |
| Ch2 `dec_e4s_c1nx`: decoder stage at inference (functional tables of every frame, estimator table, support, targets, decoder on the start-up frames, pack) | CPU only | 234 s on 72 cores (the decoder 120 s: 0.115 cpu-s per frame of the set, about 0.75 per decoded frame) |
| **Ch2 total GPU** | | **none for the inverse profiles; 0.13 GPU-h for a `syn_*`, `int_*` or `dec_*` profile (the corrector and the decoder use no GPU)** |
| **Total GPU to rebuild the submitted configuration (`hc10x24` + `dec_e4s_c1nx`) from the training entrypoints** | | **about 3.7–3.9 GH200 GPU-h, inside 24 GPU-h: Ch1 rebuild about 3.6 (measured part 3.0) plus the Ch1 metric models (not separately measured, about one teacher fit) + Final-input network about 0.07 (one job cannot exceed 0.17) + DIII-D twin 0.06; with the development selection of the `hc10x24` heads (about 2.5, not needed for a rebuild) about 6.2–6.4. GPU-hours on GH200; the factor to A100 GPU-hours was not measured. Final-input network in the Final run: 0.066 GPU-h on `cuda`, one NVIDIA GH200 120 GB (job 7538629: 238 s elapsed × 1 GPU; the training itself 157 s). For the third Final submission (`xhc` + `dec_e4s_c1nx`) the total is about 5.3–5.6 GH200 GPU-h, also inside 24 GPU-h: the row `ch1/train_xhc.sh` above (4.585, measured; it replaces the base and head rows of `train.sh`) + out-of-fold teacher statistics 0.20 + development-run heads and teacher behind the blend constants about 0.4 (estimate) + the Ch1 metric models (not separately measured) + Ch2 0.13. The entrypoint and the Final-input network alone are about 4.65** |

## 5. Tools

Software: Python 3.11, PyTorch 2.5.1, NumPy, SciPy, scikit-learn, scikit-image, pandas, pyarrow, FreeGS 0.8.2 (free-boundary
Grad–Shafranov solver and DIII-D wall), the organisers' starter kit (scorer, validator, `data_fixes.py`) at commit `a674291`.
Exact versions and hashes are in `env/`.

Use of AI assistants (the same statement as METHODS.md section 6): AI coding assistants, including agentic ones, were used
throughout this entry: to write and review code, to run and monitor experiments, to check candidate files against the challenge
rules, and to draft documentation. The entrant directed the work, decided between candidates, chose the submitted files and takes
responsibility for the submission and for the statements in the methods report and in this provenance statement.
