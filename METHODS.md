# Methods report

Submitted configuration: DIII-D profile `hc10x24` (`ch1/profiles/DEFAULT`) and MAST profile `dec_e4s_c1nx`
(`ch2/pipelines/DEFAULT`); the third Final submission pairs the DIII-D profile `xhc` (section 2, last paragraph) with the same
MAST file. The data provenance, every use of Development-board scores and the declared compute are stated in PROVENANCE.md.

## 1. Task and overall design

From coil currents and Thomson-scattering profiles only, predict per EFIT time slice the 65 × 65 poloidal flux map ψ(R, Z) and
the scalars q95 and β_N. Two different methods are used: a learned model on DIII-D, where labels exist (Challenge 1), and, on
MAST, where no label may be used (Challenge 2, zero-shot), a free-boundary Grad–Shafranov inverse fitted per frame to the
Thomson data, blended with a network trained on synthetic equilibria of our own forward model, then refined by a start-up decoder
and by a map corrector trained on DIII-D.

## 2. DIII-D: learned map with a consistency correction

**Inputs.** 789 features per frame on the EFIT time base: coil and plasma currents with boxcar / difference history, toroidal
field, Thomson T_e / n_e profiles under a validity contract, profile summaries, time descriptors. Two repairs are part of the
feature definition: (i) *mirror-canonical currents* — a reversed-I_p shot is an exact mirror of a normal one, so the 430 columns
that are odd under current reversal are multiplied by the shot's I_p sign (the raw sign is kept as a flag); (ii) an inputs-only
correction of the I_p time axis on the shots where the organisers' `data_fixes.py` keeps the shipped axis: in the older
data-acquisition era (magnetics record starting before −6 s) the I_p record is placed 8,192 ms after the start of the magnetics
record if that puts the breakdown (10 % I_p crossing) within [−100, +150] ms; otherwise a breakdown later than 40 ms on the shipped
axis is moved to 15 ms. No label is used, and the same feature code applies the rule to the training and the inference rows (rule
in PROVENANCE.md section 1).

**Targets.** ψ is compressed by PCA (80 and 120 components, fitted on training frames); q95 and β_N are regressed directly.

**Predictors.** (a) Eleven frame MLPs on the PCA coefficients (three architectures, dropout 0.3, an auxiliary head on
scorer-derived scalars) trained with a *metric loss*: a quadratic form in coefficient space obtained from the Jacobian of the
official score terms, so that coefficient errors are weighted by their effect on the composite. (b) A temporal window model
(four seeds) on nine-frame windows. (c) Seventeen scalar models in seven families for q95 and β_N. The map is the mean of (a)
and (b).

**Consistency correction.** The score's Consistency term compares seven scalars derived from the submitted map by the scorer's
own code (R_axis, Z_axis, κ, δ_top, δ_bot, V, l_i) with those of the label map. Direct heads predict these scalars (and four
LCFS extrema) from the inputs; a damped Gauss–Newton step in PCA space moves the blended map so that its scorer-derived scalars
agree with a robust fusion of the heads and the map's own values, with the metric matrix as the prior (so the step is cheap in
R²ψ). A final guard keeps l_i inside the scorer's validity window. The correction is label-free at inference.
Heads: residual MLP 1024 × 8, dropout 0.15, 24 seeds, on the same clock-corrected feature rows as the base models; capacity
chosen on a training-only validation slice; accepted over the package's earlier head set by a pre-registered paired test on a random
500-shot fold (ΔS = +0.000054 ± 0.000012), neutral on the 150-shot holdout, and kept after a Development-board comparison
(PROVENANCE.md section 2.6).

**Training protocol.** Split by shot. 150 shots are a fixed development holdout; two 500-shot folds (one random, one blocked in
time) are used for paired comparisons with shot bootstraps. The submitted weights are refits on all 7,041 labelled shots with
fixed epochs and seeds (base models 1.57–1.65 GPU-hours on one node with 4 GH200 GPUs, four clean-room runs; the 24 heads
1.35 GPU-hours on GH200 GPUs, measured in development).

**Result.** Development board (submission #14; the same DIII-D file in #15–#22): S = 0.996998 (R²ψ 0.999946, R²q95/β_N
0.9987, D_LCFS 0.007257, Consistency 0.989745).

**Profile `xhc` (third Final submission): contour fine-tune of the base members.** About half of our D_LCFS comes from frames
whose predicted contour passes the saddle at a neighbouring grid node of the scorer's marching-squares extraction. Profile `xhc`
continues the training of the eleven frame MLPs and the four temporal-window models of `hc10x24` from their weights (40 epochs,
learning rate 2e-4 → 2e-5 cosine, EMA 0.999, the members' own seeds and losses) with one added term computed on the label's
own last closed flux surface: for every vertex of the scorer's reference polygon of a training label, the first-order normal
displacement of the member's predicted contour (the member's decoded-map residual on the vertex's grid edge and at the label's
critical node, divided by the label's normal flux gradient), combined per frame by an arc-length-weighted smooth maximum with
equal weight per shot. Its multiplier is set per member to five times the metric loss's output-gradient norm (median over 16
fixed batches), capped per batch at twenty times. Training labels only. The heads, the frozen constants, the Gauss–Newton step,
the guards and q95 / β_N are those of `hc10x24`; only the map the correction starts from changes. The loss, its weight and the
length were fixed in development on the 150-shot holdout and confirmed on the two folds against the earlier base (six settings
counted on the random fold). Against `hc10x24`, a paired test written down before its evaluation (random fold ≥ +0.00004 with
the lower bound above zero, no loss on the blocked fold or the holdout beyond fixed limits, l_i and R²ψ guards) gave
ΔS = +0.000092 ± 0.000009 on the random fold, +0.000074 ± 0.000028 on the blocked fold and +0.000069 ± 0.000027 on the holdout
(1 − D_LCFS up by 0.00025–0.00049, Consistency up by 0.00013–0.00021, κ the main scalar). It was not submitted to the Development
board.

## 3. MAST: zero-shot free-boundary kinetic inverse

**Forward model.** Free-boundary Grad–Shafranov on the 65 × 65 grid with FreeGS; coils from the geometry columns shipped with
every shot; the solenoid driven with the documented factor 0.5; a public first-wall polygon as limiter. No vessel or coil-case
current is modelled.

**Closure.** The measured coil currents alone do not fix the plasma position without magnetics. The radial position is closed
by a soft iso-flux condition: the last closed flux surface passes through the Thomson outer edge (density foot) displaced by a
fitted offset, realised by a uniform vertical field with a prior on its size; the vertical position is closed by a ridge
towards the up–down balance of the measured currents.

**Kinetic inverse.** Per frame, θ = (edge offset, log pressure scale, FF′ shape coefficients) is fitted by a regularised
Levenberg–Marquardt iteration to both Thomson edges and to pairs of points on the two sides of the magnetic axis with equal T_e
(isotherms are flux surfaces) at five levels. The total plasma current is matched exactly; the pressure profile shape is the
Thomson electron pressure. Priors of the profile family come from a fit of the same family to DIII-D label maps. Frames are
solved in sequence within a shot with warm starts; ambiguous frames are re-solved from several starts and in reverse time
order; frames that fail fall back to a closed-form solve.
*l_i datum.* The Thomson data constrain the current profile weakly. A ridge model trained on DIII-D labels predicts the
cylinder-equivalent internal inductance of a frame from the dimensionless plasma-current ramp history (in resistive-time units)
and two Thomson profile exponents; it enters the fit as one Gaussian pseudo-observation of the solution's own l_i,c. The
cylinder-equivalent form is used because l_i itself does not transfer across aspect ratio. A fallback solution is shipped only
if it passes the inverse's own validity test.

**Network on our own synthetic equilibria, and blend.** For every second frame (with I_p ≥ 50 kA and a measured outer Thomson
edge) of the shots being predicted, the forward model is run with the frame's measured coil currents and plasma current and ten
draws of the profile and closure parameters (profile parameters of a cubic FF′ family with an edge term, pressure-gradient shape
s = 0.5, drawn around the medians of a fit of that family to DIII-D label maps; the closure offset from an a-priori 3 cm prior
about the frame's measured outer Thomson edge); Thomson measurements are rendered on the frame's real channels with three noise
levels. A residual MLP ensemble (512 × 4, four members) is trained on these synthetic pairs (X-point-bounded samples; about
3 minutes on one GPU) to map the input features to 48 PCA coefficients of (ψ − ψ_vacuum) / I_p and to q95, β_N; the exact vacuum
flux of each frame is added back. The network is *transductive in the test inputs and label-free*: it is trained on synthetic data
driven by the input columns of the shots being predicted, for a Final file on the Final inputs, by the network steps of
`ch2/predict_mast.sh` (in the Final run as jobs of `tools/final_slurm.sh`; the training entrypoint `ch2/train_mast_net.sh`
makes the same calls), with its GPU time counted in the compute total; the shipped network of each Final submission is the one
that its Final run trained. The blended map is w × network + (1 − w) × inverse with w = 0.5. The weight w (150 DIII-D holdout
shots, confirmed on 150 further shots) and the smoothing along the shot (1 frame) were chosen on DIII-D with a DIII-D twin of
the whole construction; the network's two scalar constants were calibrated on DIII-D training shots with the same twin.

**Map corrector trained on DIII-D.** The map is corrected in geometry, not in pixels. For every frame, the magnetic axis and four
points of the last closed flux surface (top, bottom, inboard, outboard) are extracted from our map with the scorer's published
functionals and the public first wall. A gradient-boosting model, trained on DIII-D training shots (4,104 shots outside every
holdout and evaluation list; the same blend construction run on DIII-D with the MAST-like Thomson chord), predicts from 41
dimensionless features of our own map and of the inverse's diagnostics how far the label's points lie from ours (in units of the
minor radius) and the ratio of the label's l_i to ours. On MAST the features are centred on the medians of our own maps of the
shots being predicted, the mean predicted correction over those shots is removed (DIII-D's average offset is not transferred), and
a support gate switches the correction off outside the DIII-D feature range. The map is then warped smoothly through the moved
points and its interior flux re-mapped to the corrected l_i; q95 and β_N are not changed. Strength (setting `c1nx`): the four
boundary points at half strength, l_i at full strength, no axis term, the setting that worsens none of the label-free comparisons
of our MAST maps with the Thomson data; it also scored slightly above the full-strength setting on the Development board
(PROVENANCE.md section 2.6). On DIII-D (150 holdout shots, all frames, MAST-like Thomson chord, clean / stressed) this setting
adds +0.017 / +0.022 to the S of the blend, and +0.020 / +0.026 on 150 untouched shots (on the decoded maps its gain is at least
0.99 × that). Label-free on MAST it can be checked only for safety (edge positions, symmetry centre), not for gain.

**Start-up decoder before the corrector.** The network and the inverse are two correlated estimators whose errors differ by
functional: on MAST the network is the better estimator of the axis radius and the volume, the inverse of elongation and l_i,
and in the start-up phase (plasma not yet X-point-bounded, or the current still ramping) the two disagree most. For the frames of
that phase (17 % of the public-test frames; a label-free rule on the inputs and the inverse's own solution), the blend's plasma
flux is moved along twelve smooth, current-neutral response coordinates (shifts, dilations, shear, triangularity, squareness, an
axis shift, an interior l_i re-map and the network-minus-inverse difference) by three Gauss–Newton steps until the scorer's own
functionals of the map approach per-functional targets, at minimal flux change. Each target is a convex combination of the two
estimators' functionals with weights from (i) their error moments on the DIII-D training pool and (ii) a min–max rule on our
own synthetic MAST-geometry equilibria driven by MAST input frames, mixed by a support gate that trusts the DIII-D weights only
where the frame's features lie inside the DIII-D range. Prior widths and the trust region come from DIII-D training errors; the
target scales and the flux normalisation are statistics of our own maps of the shots being predicted. All other frames keep the
blend map; then the corrector (setting `c1nx`) is applied, its centring and zero mean recomputed on the decoded maps. On DIII-D
the decoder is a tie with the corrected blend (start-up frames are few there); on synthetic MAST-geometry truth it gains
+0.003 to +0.005 in S (four populations, 90 % lower bounds above zero).

**Scalars.** q95 and β_N are those of the blend (weighted geometric mean for q95, weighted mean for β_N, with the blend weight;
then a validity window and a centred 3-frame running median along the shot) and are not changed by the decoder or the
corrector. The inverse's q95 = F(ψ95) / 2π ∮ dl / (R |∇ψ|) on its map and its β_N = c × β_N,e (Thomson electron pressure on its
flux surfaces), and the network's q95 and β_N heads, each carry one constant, calibrated on DIII-D training shots with DIII-D's
Thomson data re-sampled on a MAST-like horizontal chord, so that the calibration sees MAST-like coverage (the network's constants
with its DIII-D twin).

**What transfers from DIII-D.** Scalar constants, the prior means of the profile families (and part of their widths), four
closure constants, the l_i model, the blend weight and the network's smoothing width, the map corrector, and the decoder's source
weights, support table and prior widths (PROVENANCE.md section 2.3). Everything else is a-priori settings, geometry, public machine
description and MAST input columns: of the shots being predicted and, in development, label-free statistics and checks on MAST
inputs (the decoder's synthetic-MAST weights from every third public-test shot, the network's feature-dropout rates, the design
checks of the inverse; PROVENANCE.md sections 2.2, 2.4); choices between complete candidates also used Development-board scores
(next paragraph). A public NSTX-U equilibrium set (Wai, Boyer and Kolemen, Nucl. Fusion 62 (2022) 086042) was analysed as an
aspect-ratio check and read as supporting evidence when the l_i-datum pipeline was put on the Development board; no constant,
weight, prior or blend weight was fitted on it and no network was trained on it (PROVENANCE.md section 2.5).

**Use of Development-board feedback.** The scores and Detailed Results that the Development phase returned for our own submissions
were used to choose between complete, separately submitted candidates; no constant was fitted to them (full list: PROVENANCE.md
sections 2.6–2.7). For the submitted MAST configuration this concerns the numerical precision (float32), the solenoid factor (the
documented 0.5; the switch from 1.0 was motivated by the board result), the inverse variants that led to `p7li`, the blend weight
(0.5 over 0 and 1), the corrector strength (`c1nx` over full strength), the start-up decoder (kept by a rule written before its
score was read) and its restriction to the start-up frames (decided after per-term results had been seen), and the scalars (the
blend's, over the rules of the option `dec_e4s_c1nx_a1w5`, which is not submitted); further variants were not adopted. The per-shot
detail of one submission was used in an analysis of the MAST error. On DIII-D, `hc10x24` was kept over `fccx` (section 2).

**Result.** Development board (submission #17; #21 is an identical re-upload with the same scores): MAST S = 0.7601
(R²ψ 0.9812, R²q95/β_N 0.4501, D_LCFS 0.1720, Consistency 0.3504); with the DIII-D file above, G = S(MAST) / S(DIII-D) = 0.7623
(as shown on the board).

## 4. Numerical reproducibility and compute

- DIII-D file: regenerated from raw parquet with the shipped weights in a clean room (environments from the lock files only): same
  md5 as the Development file on the reference hardware. Retraining from the listed generated data reproduces the base models in
  clean rooms (sha256-identical; the 4 temporal-model checkpoints tensor-identical) and, once in development with the same
  commands, the 24 heads (sha256-identical); the heads' retrain has not been repeated through the package entrypoint in a clean
  room.
- MAST file: the dress rehearsal of the Final procedure computed every stage from the public-test inputs (inverse solve, synthetic
  set, the network trained afresh, blend, decoder, corrector, statistics over the shots) and gave the md5 of the Development file
  #17 on the reference hardware, with network weights equal to the Development network's (generation is seeded per shot). Another
  BLAS kernel is not bit-identical: the inverse is an iterative non-linear fit with warm starts and discrete branch decisions, so
  last-bit differences grow. For the plain inverse (`p7a`) with OpenBLAS's generic ARMv8 kernels only 4.5 % of the frames stay
  bit-identical (median per-frame maximum deviation 1.2 % of the flux span), and on DIII-D, where labels exist, the composite
  moves by +0.00012; the effect on the submitted chain was not measured. The network that the Final run trained is published
  (PROVENANCE.md section 3), so the MAST file can be regenerated without retraining.
- Compute: rebuilding the submitted configuration from the training entrypoints takes about 3.7–3.9 GH200 GPU-hours (Ch1 about
  3.6, of which 3.0 measured: base models 1.65, heads 1.35; plus the Ch1 metric models, not separately measured; the Final-input
  network about 0.07; the DIII-D twin network 0.06), about 6.2–6.4 with the development selection of the head hyper-parameters,
  against the rules' limit of 24 GPU-hours on a single 80 GB A100-class GPU or equivalent; the factor between GH200 and A100
  GPU-hours was not measured. The Final run's own value for the network is recorded in PROVENANCE.md section 4, as are the CPU-only
  data generation (including the synthetic set on the Final inputs), the inverse, the corrector (trained on CPUs with LightGBM) and
  the decoder (no GPU, about 0.75 cpu-seconds per decoded frame). From the inverse and network outputs, the decoder and corrector
  stages reproduce the whole MAST file bit for bit.

## 5. Limitations

- The MAST label maps are magnetics-only reconstructions that include vessel and coil-case currents and use a two-parameter
  profile family; a Thomson-based inverse without those currents cannot match them frame by frame, and its shape scalars are
  prior-dominated in the directions the Thomson data do not constrain.
- The network inherits the forward model's assumptions; it reduces the frame-to-frame variance of the inverse rather than adding
  information the forward model does not have.
- The corrector learns how DIII-D labels differ from our DIII-D maps; whether MAST labels differ from our MAST maps in the same
  way cannot be tested without MAST labels. The zero-mean step and the support gate limit what is transferred to the shape of the
  correction, not its average.
- The decoder's synthetic weights come from our own forward model, which cannot represent the vessel currents of the label
  reconstructions; its DIII-D evidence is a tie, its case for MAST rests on synthetic MAST-geometry truth and label-free Thomson
  checks, and the restriction to the start-up phase was decided after Development results were seen (PROVENANCE.md
  section 2.6).
- Data provenance per machine, every use of Development-leaderboard scores and an input-only matching of the public-test MAST
  inputs against the public archive in September 2026, whose only trace in the pipelines of submitted files, a geometry-fetch
  guard that changed no value, is described in PROVENANCE.md (sections 1, 2.1–2.7).

## 6. Use of AI assistants

AI coding assistants, including agentic ones, were used throughout this entry: to write and review code, to run and monitor
experiments, to check candidate files against the challenge rules, and to draft documentation. The entrant directed the work,
decided between candidates, chose the submitted files and takes responsibility for the submission and for the statements in this
report and in the provenance statement.
