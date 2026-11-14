# Final-phase runbook (2026-10-19 00:00 UTC → 2026-10-26 00:00 UTC)

_Operational checklist for the entrant. Nothing in it is run automatically. Steps marked **[decision]** need the entrant.
Wall-clock figures are measurements of the clean-room rehearsals of 12–15 October 2026 on the Development configurations (nodes
with 288 Arm cores and 4 GPUs); where a figure was not measured, it says so. Section B was replayed end to end on 14–15 October
(dress rehearsal, section B.9): both files bit-identical to the Development files, 2 h 32 min from the start to both files
validated, 65,730 allocated cpu-minutes = 3.80 node-hours._

Final inputs: Hugging Face dataset `Sophelio/fusion-equilibrium-challenge-final`, configurations `diii_d_private_test` (879 shots)
and `mast_private_test` (1,205 shots), published 2026-10-19 00:00 UTC. Output file names: `diii_d_private_test.npz` and
`mast_private_test.npz`. Three Final submissions; by default the most recent scored one counts; a nomination email
(fusionec26@sophelio.io, team name + two submission IDs) can choose otherwise before the close. **Every Final submission carries the
reproduction package (pinned commit or archive sha256) in the fact sheet's Code URL field**, including the trained weights used
for that submission.

There is no time pressure on 19 October: the phase is seven days long and one complete run takes about two and a half hours.
Robustness matters, speed does not.

## A. Before 19 October (freeze)

1. **[decision]** DIII-D profile: `fccx` or `hc10x24` → write the name into `ch1/profiles/DEFAULT` (`ch1/profiles/README.md`;
   clean-room status of each in PROVENANCE.md section 3). The validation folds favour `hc10x24` (PROVENANCE.md section 1); `fccx`
   is the fallback. Always run a profile through `ch1/predict.sh` — it hands the profile's `HEADS` setting to the inner script,
   whose own default is the `hc10x24` heads.
2. **[decision]** MAST profile → `ch2/pipelines/DEFAULT`: `p7a`, `p7c`, `p7a_tsw05`, `p7li`, `syn_xa_p7li50`, `syn_xa_p7c80`,
   `int_b50c1nx`, `int_b50gld`, `dec_e4s_c1nx`, `dec_e4s_c1nx_a1w5`, or a new profile added by the drop-in procedure in `ch2/pipelines/README.md`. The `int_*`
   and `dec_*` profiles need `src/.vendor_scorer` (built by `env/build_env.sh`) and the files of `weights/MANIFEST_CH2.sha256` (corrector
   model; for `dec_*` also the decoder's support source table; fetched by default). A profile may be used for a Final file only after a
   clean-room run has reproduced its Development-phase file (same md5): done as whole files for `int_b50c1nx`, `int_b50gld`,
   `dec_e4s_c1nx` (Development submission #17) and
   `dec_e4s_c1nx_a1w5` (its maps with the scalar rules A1 / B7w5, md5 `4d66f8fc…`) are reproduced as whole files (PROVENANCE.md
   section 3); their Final-day sequence, checks and cost are in section B.8.
3. **[decision]** If the MAST profile is a `syn_*`, `int_*` or `dec_*` profile (the `int_*` and `dec_*` profiles contain the `syn_xa_p7li50` network): accept that it is *transductive in the test inputs* (its network is
   trained during the Final run on synthetic equilibria driven by the Final input columns; PROVENANCE.md section 2.2) and that the
   package commit named in the fact sheet must therefore be made AFTER the run (step B.7). If that is not wanted, choose an inverse
   profile.
4. **[decision]** Wording of METHODS.md section 6 (three draft options) and PROVENANCE.md section 5; re-read PROVENANCE.md against
   the chosen profiles (constants tables, synthetic data, other machines' data, Development-board uses).
5. **[decision]** Publication route: public Hugging Face model repository (code + `env/` + `weights/`) and a code mirror. Record the
   40-character commit SHA and the sha256 of every weights archive. For an inverse MAST profile the package can be frozen and
   published before 19 October; for a `syn_*` profile publish the frozen tree now and add the Final-run network in one further
   commit (step B.7).
6. Rehearse the public fetch once: fresh directory, `bash env/build_env.sh`, `bash tools/fetch_weights.sh hf <repo> <sha>` (by
   default it fetches and verifies every weight file the selected profiles need: the manifests of the DIII-D profile — for `hc10x24`
   `weights/MANIFEST.sha256` and `weights/MANIFEST_CH1_HC10X24.sha256` — and `weights/MANIFEST_CH2.sha256`; it prints `WEIGHTS_OK`
   only if all of them are present with their sha256; with archives (`url` mode) give `MANIFESTS=` per archive), then the two
   predict entrypoints on a few public-test shots, compare with the clean-room hashes. On Final day the weights must be placed in
   the NEW work directory (`FEC_WORK`) of the run: `tools/final_slurm.sh submit` checks them against the manifests before it submits
   anything (section B).
7. Compute and storage for the Final days:
   - one complete run (both machines) is BILLED 65,730 cpu-minutes = 3.80 node-hours with `hc10x24` + `dec_e4s_c1nx` (dress
     rehearsal, section B.9; 1 node-hour = 288 cores x 1 h = 17,280 cpu-minutes; the allocation portal bills each job by the
     largest of its core, GPU and memory shares of a node and ran about 1.3 % ahead of the scheduler's counter) and about 41,000
     cpu-minutes with an inverse profile (table "CPU-minute budget of the Final submissions" in section B; the programs themselves
     use about 700 / 350 core-hours, the rest is the 72-core billing floor and idle cores), and less than 0.2 GPU-hours; keep at
     least three runs' worth of the project allocation free;
   - the CPU path (`DEVICE=cpu`, GPU-wait rule in section B) costs about 63,800 cpu-minutes but about **3.89 node-hours** per run:
     its DIII-D job runs on 72 cores for about 6,700 s (8,023 cpu-minutes measured on 14 October) and requests 160 GB, the measured
     working request and the only job above 100 GB per 72 cores; its memory share of a node (160 / 449 GB = 0.36) exceeds its core
     share (0.25), so the allocation bills it about 42 % above its cores (about 0.66 instead of 0.46 node-hours); its network
     training takes about 15 minutes on 72 cores (about 1,300 cpu-minutes, 0.08 node-hours) instead of 4 minutes on one GPU;
   - on 13 October the project's CPU-minute limit (`AssocGrpCPUMinutesLimit`) stopped jobs from starting while many other jobs of
     the account were queued: on Final day run nothing else on the account and give every job a tight time limit (the scheduler
     books cores × time limit; one run books up to about 117,000 cpu-minutes at a time, table "Three Final submissions" in
     section B);
   - work file system: about 17,000 files and 60 GB (dress rehearsal: inputs 10.4 GB for 2,080 shots, DIII-D run 31 GB, MAST run
     19 GB incl. the synthetic set; weights 3 GB). Check the inode and byte quota the day before (the inode quota was exhausted once on 12 October).
8. A Hugging Face READ token for the run (never printed, never committed) even though the Development dataset is public:
   unauthenticated streaming from one address was rate-limited (HTTP 429, retries of 2–3 minutes) on 13 October when several
   validators ran at once. The driver reads it inside the jobs from `HF_TOKEN_FILE`, by default in this runbook
   `HF_TOKEN_FILE=$HOME/.config/fec/hf_read_token` (a file of mode 600 holding a scoped read token; check its presence with
   `test -s $HF_TOKEN_FILE && echo present`, never print it); for the plain commands of section B export `HF_TOKEN` instead.
   Rotate any token that has ever been printed into a log.

## B. Production run

Environment: the frozen package checkout with its two environments; `FEC_WORK` on the work file system. `<P>` = the MAST profile,
`<C>` = the DIII-D profile. Commands are shown as plain commands; on a cluster each numbered block is one job (or one job per
shard). Run the Hub-streaming steps (B.2, B.6) one at a time.

**On a SLURM cluster: `tools/final_slurm.sh`** submits exactly the commands below as jobs with dependencies (one call; job shapes at
or below 100 GB per 72 cores, time limits about 1.5 x the measured run times, the `init` step before the shard jobs, the two Hub
streams and the two validators one at a time), then checks the run against the table "Checks before anything is uploaded" and
reports the cost of every job. Before it submits anything, `submit` checks the environments, the starter kit, `src/.vendor_scorer`
(`int_*` / `dec_*` profiles) and every weight file of the selected profiles against its manifest in `FEC_WORK` (lines `PREFLIGHT …`;
`PREFLIGHT_FAILED: nothing submitted` otherwise):
```bash
export FEC_WORK=/path/to/work/final HF_TOKEN_FILE=$HOME/.config/fec/hf_read_token; cd <package checkout>
bash tools/final_slurm.sh submit          # DEVICE=cpu for the CPU fallback; STEPS="..." to (re)submit single steps; DRYRUN=1 to print
bash tools/final_slurm.sh status          # per job: state, elapsed, allocated cpu-minutes, node-hours
bash tools/final_slurm.sh check           # PASS / FAIL per check; md5 of both files
```

**Re-submitting a step that already started in the same run** (`STEPS=…` with the same `RUN`): the DIII-D predict refuses an
existing run directory and the MAST finish refuses to overwrite its packed / post / submission files, so such a job would fail
after waiting for its turn. `submit` therefore stops with `STARTED_BEFORE: …` when `STEPS` contains `ch1` and the DIII-D run
directory exists, or `finish` and a MAST output file of the run exists. `REDO=1` moves those outputs aside first (nothing is
deleted): `$FEC_WORK/stage11_ch1/final_path/runs/<RUN>` → `<RUN>.redo_<UTC time>`; `$FEC_WORK/ch2_runs/<RUN>/post` and
`…/submission` → `post.redo_<UTC time>`, `submission.redo_<UTC time>`. The manual equivalent is the same three `mv` commands.
Example after a DIII-D job that started and failed: `REDO=1 STEPS="ch1 val_d3d" bash tools/final_slurm.sh submit`; after a failed
MAST finish: `REDO=1 STEPS="finish val_mast" bash tools/final_slurm.sh submit` (merge, the β_N table and every post stage recompute
their outputs; the solve, synthetic-generation and network outputs are kept). The GPU→CPU switch below re-submits only jobs that
never started and needs no `REDO`.

**GPU-wait rule.** Probe first (one tiny job: `sbatch -N 1 --gpus=1 --cpus-per-task=8 --mem=16G -t 5 --wrap "nvidia-smi -L"`):
if it starts within 3 hours, submit with `DEVICE=gpu` (dress rehearsal: started after 1 s; on 14 October a 1-GPU job waited
2 h 22 min). If the run's two GPU jobs (`ch1`, `syntrain`) have still not started 3 hours after submission (and in any case by
10-19 06:00 UTC), cancel them and the jobs that depend on them and resubmit those steps on the CPU, keeping the CPU stages that
already finished: `DEVICE=cpu STEPS="syntrain ch1 finish val_d3d val_mast" bash tools/final_slurm.sh submit`. The CPU path was
rehearsed on 14 October: valid files, not bit-identical to the GPU ones (DIII-D: 150 labelled holdout shots, composite +0.0000038 ±
0.0000026; network: a change of the size of a seed replica); DIII-D 6,677 s on 72 cores, network training 902 s. Decide GPU or CPU
once, before the first upload: the DIII-D file must be the same in all three submissions.

```bash
export FEC_WORK=/path/to/work/final; cd <package checkout>
PY=src/.venv/bin/python
REPO=Sophelio/fusion-equilibrium-challenge-final; D3D=diii_d_private_test; MAST=mast_private_test     # the Final
#   dress rehearsal on the public test: REPO=Sophelio/fusion-equilibrium-challenge; D3D=diii_d_public_test; MAST=mast_public_test

# B.1 inputs (one parquet file per shot)                                              [6 min; 10.4 GB for the public test]
$PY tools/download_inputs.py --repo $REPO --config $D3D --out $FEC_WORK/in/d3d
$PY tools/download_inputs.py --repo $REPO --config $MAST   --out $FEC_WORK/in/mast
ls $FEC_WORK/in/d3d/*.parquet | wc -l     # expect 879 (public test: 874)
ls $FEC_WORK/in/mast/*.parquet | wc -l    # expect 1,205 (public test: 1,206)
#     Files are taken from data/<config>/; if that folder is empty, from the configuration's data_files pattern in the dataset card.

# B.2 stream order of each configuration (the scorer indexes shots by the streamed order)   [41 min DIII-D, 36 min MAST (dress rehearsal, no token)]
$PY tools/stream_order.py --shot-dir $FEC_WORK/in/d3d  --repo $REPO --config $D3D --out-dir $FEC_WORK/order/d3d
$PY tools/stream_order.py --shot-dir $FEC_WORK/in/mast --repo $REPO --config $MAST   --out-dir $FEC_WORK/order/mast
#     ORDER_CHECK OK required (exit code 1 -> stop). Always pass the order file, never "sorted".

# B.3 MAST, CPU stages - start them right after B.1, they do not need the order          [max 14.5 min per solve shard, 32 jobs x 72 cores]
PIPE=<P> NSHARD=32 STAGES=init bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz        # run directory + shard lists, ONCE, before the shard jobs
for k in $(seq 0 31); do PIPE=<P> NSHARD=32 SHARD=$k STAGES=solve bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz; done    # one job per shard
#     syn_*, int_* and dec_* profiles only, in parallel with the solve shards:
for k in $(seq 0 31); do PIPE=<P> NSHARD=32 SHARD=$k STAGES=syngen bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz; done   # one job per shard [12 min]
PIPE=<P> NSHARD=32 STAGES=synreal bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz                                          # [15 s]

# B.4 syn_*, int_* and dec_* profiles only: one GPU - train the network on the synthetic set and predict every frame   [4 min; training 3 min]
PIPE=<P> NSHARD=32 STAGES=syntrain bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz
#     init + syngen + synreal (B.3) + syntrain (B.4) are the steps of the training entrypoint of the Final-input network,
#     ch2/train_mast_net.sh (the same calls, one after the other on one machine; added after the Final checkout 82e50ff, whose run
#     makes these calls as the jobs of tools/final_slurm.sh); its GPU time is this job's (PROVENANCE.md section 4)

# B.5 after B.2: DIII-D (one node: all cores + 1 GPU)                                   [36 min predict + 3 min gate]
CH1_PROFILE=<C> bash ch1/predict.sh $FEC_WORK/in/d3d final $FEC_WORK/order/d3d/stream_order.txt $D3D.npz
#     -> $FEC_WORK/stage11_ch1/final_path/runs/final/$D3D.npz, gate.json / gate.log, profile.txt
#     The log must contain, in this order: the header line with "HEADS=<v21|v22c10>", for hc10x24 the stage line
#     "heads v2.2 c10 (24 seeds)", "HEADS_CHECK OK: profile <C> ...", "ALL_DONE"; and must NOT contain "PREDICT_INCOMPLETE".
#     MAST finish: merge, beta_N table, pack, post stages (blend for syn_*; blend + corrector for int_*; blend + start-up decoder +
#     corrector for dec_e4s_c1nx), stream order, local format check (shot i of the file against the input file at stream index i
#     of the order file)                                                               [3.5 min; int_*: + at most 4 min; dec_*: section B.8]
PIPE=<P> NSHARD=32 STAGES="merge scal pack post validate" bash ch2/predict_mast.sh $FEC_WORK/in/mast final $FEC_WORK/order/mast/stream_order.txt $MAST.npz
#     -> $FEC_WORK/ch2_runs/final/submission/$MAST.npz, md5.txt, sha256.txt, record.json

# B.6 the organisers' validator (unmodified) on the Final configurations, one after the other   [23 min + 18 min]
$PY tools/validate_final.py $FEC_WORK/stage11_ch1/final_path/runs/final/$D3D.npz --repo $REPO --config $D3D
$PY tools/validate_final.py $FEC_WORK/ch2_runs/final/submission/$MAST.npz           --repo $REPO --config $MAST

# B.7 syn_*, int_* and dec_* profiles only: publish the network this run trained, check it through the inference entrypoint
#     (MANDATORY), then pin the package. F = the run of the submitted MAST file: final, or final_cpu after the CPU fallback.
F=$FEC_WORK/ch2_runs/<final|final_cpu>; V=${F}_shipped     # V must not exist yet
$PY tools/publish_model.py $F/syn/models/model_mCxa32.pt <publish dir>/stage11_ch2/final/model_mCxa32.pt
#     -> prints both sha256 (original = "run_models" of the run's submission/record.json; published copy). Fill in PROVENANCE.md:
#        section 3 <FINAL_NET_SHA256_ORIG>, <FINAL_NET_SHA256_PUB> and <FINAL_OPENBLAS_KERNEL> (the kernel OpenBLAS chose on a node
#        of the run, e.g. threadpoolctl in the ML environment); section 4 <FINAL_SYNTRAIN_GPU_H> (sacct elapsed x gres/gpu of the
#        syntrain job; 0 after the CPU fallback), <FINAL_SYNTRAIN_DEVICE> ("device" in $F/syn/models/train_mCxa32.json) and
#        <FINAL_SYNGEN_WALL> (first start to last end of the syngen jobs, tools/final_slurm.sh status). Append "<bytes> <sha256 of
#        the published copy> stage11_ch2/final/model_mCxa32.pt" to weights/MANIFEST_CH2.sha256, upload the file with the other
#        weights, commit, and use THAT commit SHA in the fact sheet. A further Final submission with a network of its own gets its
#        own path and manifest line.
#     Shipped-weights check, MANDATORY before the MAST file is uploaded (the organisers' rebuild check 2 runs the shipped weights
#     through the inference entrypoint): the PUBLISHED copy, through ch2/predict_mast.sh, must reproduce the Final MAST file byte for
#     byte. One job on the device of the run's network step ("device" in $F/syn/models/train_mCxa32.json): cuda -> 72 cores + 1 GPU;
#     cpu (CPU fallback) -> 72 cores, no GPU, CUDA_VISIBLE_DEVICES= (as the run's syntrain job; on the CPU the network threads are
#     the job's cores). Time limit 20 min with a GPU, 30 min on the CPU (bookings up to 1,440 / 2,160 cpu-minutes; needs a compute
#     decision). A second run directory gets HARD LINKS to what the check only reads - the solve shards with the shot lists, the
#     network input features and the synthetic set - and nothing else: the stages of the check write only new files or new links
#     (merge links the shard parts again; no stage rewrites a part file in place). Never hard-link the logs, the merged parts /
#     tables, syn/models, syn/pred, the post stages' directories (syn/blend, dec, corr, post) or submission: a stage that rewrote a
#     linked file in place would change the Final run's copy.
mkdir -p $V/syn && cp -al $F/shards $V/shards && cp -al $F/syn/real $F/syn/synth $V/syn/
SYN_MODEL_FILE=<publish dir>/stage11_ch2/final/model_mCxa32.pt PIPE=<P> NSHARD=32 STAGES="syntrain merge scal pack post validate" \
    bash ch2/predict_mast.sh $FEC_WORK/in/mast $(basename $V) $FEC_WORK/order/mast/stream_order.txt $MAST.npz
#     (CPU fallback: the same command with CUDA_VISIBLE_DEVICES= in front)
#     -> "syntrain: model taken from <published copy> (not retrained)", "model <sha256 of the published copy> ...", the PREDICT line
#        with every shot, MERGE_OK, validate_local rc=0, CH2_PREDICT_DONE
md5sum $V/submission/$MAST.npz $F/submission/$MAST.npz     # the two md5 must be EQUAL ($F/submission/$MAST.npz = the MAST file of F1)
#     Not equal: do not upload a file under that commit; keep both run directories and compare the intermediates in order (network
#     predictions syn/pred, blend syn/blend, decoded dec/run, corrected corr/out) to find the first difference.
```

Order and elapsed time of one complete run (dress rehearsal of 14–15 October through `tools/final_slurm.sh`, public test,
unauthenticated Hub; job start delays included — every job started within 10 s of its dependencies):

| elapsed | running |
|---|---|
| 0:00 | all 74 jobs submitted; B.1 download (6 min) |
| + 0:06 | B.2 DIII-D stream order (41 min); B.3 MAST `init`, 32 solve shards (max 14.5 min), 32 synthetic-generation shards (max 12 min), `synreal` |
| + 0:19 | B.4 network training + prediction on 1 GPU (4.3 min) |
| + 0:47 | B.2 MAST stream order (36 min); B.5 DIII-D predict on 288 cores + 1 GPU (38 min including the gate) |
| + 1:24 | B.5 MAST finish (9 min: merge, β_N table, pack, blend, start-up decoder, corrector, local check) |
| + 1:25 | B.6 DIII-D validator (34 min), then MAST validator (33 min) |
| + 2:32 | both files validated; B.7 and the checks below (`tools/final_slurm.sh check`: 24 checks) |

The critical path is download → DIII-D stream order → DIII-D predict → the two validators; the MAST chain finishes about 1 h
earlier. Streaming from the Hub was slower than on 13 October (stream orders 41 + 36 min instead of 39 + 21, validators 34 + 33
instead of 23 + 18): the time limits of the driver allow for that; a read token helps.

Compute of one run: MAST solve 365 core-hours of solver time, synthetic generation 333 core-hours, DIII-D predict about
150 core-hours, GPU busy for less than 10 minutes in total.

Checks before anything is uploaded (all must hold):

| check | expected | if not |
|---|---|---|
| shot counts | 879 / 1,205 parquet files, 879 / 1,205 shots in the files | stop: incomplete download |
| stream order | `ORDER_CHECK OK` for both configurations | stop |
| DIII-D profile | `profile.txt` = the frozen `<C>`; log line `HEADS_CHECK OK: profile <C>, HEADS=…`; for `hc10x24` also the stage line `heads v2.2 c10 (24 seeds)` | stop: the file was made by the other recipe (a development copy of the predict script once submitted the `fccx` recipe under the `hc10x24` name with a normal-looking log; the package entrypoint cannot do that, the check is kept anyway) |
| DIII-D run complete | log ends with `ALL_DONE` after the gate; no `PREDICT_INCOMPLETE` | run again under a new run name |
| DIII-D gate (`gate.log`) | 0 FAIL; G5 order PASS; REVIEW allowed for G7 (sign class 2) and G9 (fallback statistics), as on the public test | G7 class-3 sign or G8 off-schema record: **[decision]** read `gate.json`; the file is written with the pre-registered default (window-majority sign) |
| DIII-D derivation | `G6 zero derivation failures: PASS` | exit code 3 = residual failure, file still written: **[decision]** |
| MAST shards | every shard complete; merge prints `MERGE_OK 1205 shots` | `SOLVE_INCOMPLETE` / `SYNGEN_INCOMPLETE`: run the same shard command again (finished shots are kept). A shot that fails twice: **[decision]**, no automatic fallback exists (0 such shots in every public-test and DIII-D run) |
| network (`syn_*`, `int_*`) | stage prints the model sha256 and `PREDICT … "n_shots": 1205`; the blend prints its md5 | `syntrain: … incomplete`: run the missing stage; a failed training: run the same command again |
| corrector (`int_*`, `dec_*`) | the log of the finish job has a `KTAB mast mbl: <all frames>` line (`dec_*`: `KTAB mast dec: <all frames>`) and a `KAPPLY` line with `"n": <all frames>`; it must NOT contain `CORR_SHOTS selects`; `corr/out/apply_corr_mast.json` records the gate statistics (public test: mean 0.998) | `corr: ...` error: stop; a missing `src/.vendor_scorer` or model: `env/build_env.sh`, `tools/fetch_weights.sh` |
| start-up decoder (`dec_*`) | section B.8, table "checks of the decoder stage" | `dec: ...` error: stop |
| MAST local format check | `validate_local` with the run's stream order file: shot i of the file has the frame count of the input file at stream index i (the finish job's own `validate_local rc=0` line and the `check` row "MAST local format check (stream order)") | stop: the file is not in stream order or incomplete |
| validators | both `✓ OK`, rc 0 | HTTP 429 lines: wait and run again, one validator at a time; anything else: stop |
| record | `record.json` = the same record recomputed from the frozen checkout, every sha256 compared (`check` row "record.json = frozen package"): profile, profile sources, constants, solver configurations, shipped models (also equal to `weights/MANIFEST_CH2.sha256`) and, for `syn_*` / `int_*` / `dec_*`, the run's own network (`run_models`); for `dec_*` the corrector model and `support_source_d3d.parquet` in `models`, `wD_L.json`, `wS_L.json`, `decoder_E4.json` in `constants` | stop: wrong checkout or wrong profile |

Known operational faults and what the package does about them:

| fault (seen in rehearsal) | effect | handling |
|---|---|---|
| the scheduler's per-user node-local `TMPDIR` is removed while a job is still running (another job of the same user ended on the node) | a GPU training died with `FileNotFoundError` on a temporary directory | every MAST entrypoint uses a private temporary directory (`FEC_TMP_ROOT`, default `/tmp`); a failed stage exits non-zero; run the command again |
| Hub rate limit (HTTP 429) with several unauthenticated streams from one address; slow or timed-out reads | validators and stream-order runs wait 2–3 min per retry or fail; on 15 October one read time-out (retried at once) and streams up to 1.7 x slower than on 13 October | `HF_TOKEN` (`HF_TOKEN_FILE` for the driver); one streaming job at a time (the driver serialises them); time limits 55–75 min |
| inode quota of the work file system | shards lose shots with `Errno 122` | check the quota first; re-run the shard (resumes) |
| project CPU-minute limit | jobs stay pending | nothing else on the account; tight time limits; the budget table below |
| no GPU job starts for hours (14 October: several hundred nodes down, the idle GPU nodes held by a reservation; CPU jobs still started) | the DIII-D predict and the network training wait | the GPU-wait rule above: probe; after 3 h without a start, the two GPU steps on the CPU (`DEVICE=cpu`, rehearsed 14 October). `ch1/predict.sh` detects the device when `DEV` is not set (a GPU visible to torch → GPU path, none → CPU path) and stops if `DEV=cuda` is set without a visible GPU; the network stage uses all cores of the job when no GPU is visible (`SYN_THREADS` to override) |
| a step that started fails and is re-submitted in the same run | the DIII-D predict refuses the existing run directory, the MAST finish refuses to overwrite its files | `REDO=1 STEPS="…"` (paragraph after the driver commands): the outputs are moved aside, never deleted |
| a file of the checkout is replaced or edited while a job executes it (checkout on a network file system) | the running shell script loses its file handle and the job ends **silently, with exit status 0 and without an output file** at its next step (two predict jobs were lost this way on 13 October) | freeze the checkout before the first job and do not touch it until the last job has ended; `ch1/predict.sh` prints `PREDICT_INCOMPLETE` and exits 1 when the inner script ends without a finished file; for the MAST stages check the `CH2_PREDICT_DONE` line and the md5 line |

### CPU-minute budget of the Final submissions

The scheduler of the rehearsal cluster bills allocated cores × elapsed time, **bills at least 72 cores per job** however few are
requested, and books cores × time limit against the project limit while a job runs (so the time limit itself costs allocation
until the job ends). Figures are allocated cpu-minutes measured in the clean-room rehearsals of 13–14 October on the public-test
configurations (874 DIII-D shots, 1,206 MAST shots; the Final has 879 / 1,205, i.e. the same within 1 %). Job names in brackets.

**One Final submission, DIII-D profile `hc10x24` + MAST profile `int_b50gld` or `int_b50c1nx`** (for `syn_xa_p7li50` drop the
corrector row; for `p7li` also drop the network rows and the blend):

| step | job shape | measured | allocated cpu-min | time limit to request |
|---|---|---|---|---|
| B.1 input download (2,084 shots, about 25 GB) | 1 job, 72 cores, network-bound | not measured | about 500–1,500 (estimate, 5–20 min) | 30 min |
| B.2 stream order, both configurations | 1 job, 72 cores, the two streams in parallel (`HF_TOKEN`) | 39 + 21 min, single-threaded | 2,800 (4,300 as two jobs one after the other) | 60 min |
| B.5 DIII-D predict + gate, `hc10x24` | 1 job, 288 cores + 1 GPU | 2,679 s | 12,860 [k6_c1p_hcR] | 70 min |
| B.3 MAST inverse solve (`p7li`) | 32 jobs, 72 cores (the driver; 40 requested = 72 billed in the earlier rehearsal) | 591 s mean, 883 s max per shard | 22,700 [k5_p7li_s*] | 22 min per shard (driver) |
| B.3 the same packed into 4 jobs × 72 cores (alternative) | 4 jobs, 72 cores | not measured (206 core-h of solver CPU) | about 13,000–16,000 (estimate) | 80 min |
| B.3 synthetic generation (`syn_*`, `int_*`, `dec_*`) | 32 jobs, 72 cores | 533 s mean, 728 s max per shard | 20,500 [k5_syn_xa_p7li50_g*] | 18 min per shard (driver) |
| B.3 network inputs (`synreal`) | 1 job, 72 cores | 18 s | 20 | 5 min |
| B.4 network training + prediction of every frame | 1 job, 72 cores + 1 GPU | 251 s | 300 [k5_syn_xa_p7li50_tr2] | 7 min |
| B.5 MAST finish: merge, β_N table, pack the inverse file, blend, pack | 1 job, 72 cores | 219 s | 260 [k6_p7c80_reuse] | 10 min (with the next row) |
| B.5 corrector stage `corr` (`int_*` only, same job): table of ALL Final frames → feature centring and zero mean recomputed on the Final inputs, apply, pack | same job | rehearsal of 14 October (re-computed blend + table of all 69,075 frames + apply on 25 shots + pack): 195–203 s on 72 cores; the development build of the whole file (with extra diagnostics): 457 s | at most 550 | (included above) |
| B.5 start-up decoder stage `dec` (`dec_e4s_c1nx` only, same job, between the blend and the corrector): functional tables of ALL Final frames, estimator table, support gate, targets, decoder on the start-up frames, pack | same job | 234 s on 72 cores (clean room 8, 14 October: tables 56 s, estimator / support / targets 11 s, decoder 120 s, pack 47 s); the whole finish job of that profile (pack, blend, `dec`, `corr`, local check) 507 s | about 280 (the whole finish job about 610) | finish job 15 min |
| B.6 organisers' validator, both files | 1 job, 72 cores, the two in parallel (`HF_TOKEN`) | 23 + 18 min | 1,700 (2,900 as two jobs one after the other) | 45 min |
| **one complete submission** | | | **about 63,000 as rehearsed; about 54,000 with the 4-job solve (`dec_e4s_c1nx`: + about 350)** | |
| **dress rehearsal 14–15 October (`hc10x24` + `dec_e4s_c1nx`, `tools/final_slurm.sh`, every job ≤ 100 GB per 72 cores)** | 74 jobs | 2 h 32 min | **65,730 = 3.80 node-hours** (section B.9) | as in the driver |

The corrector's corpus statistics are recomputed automatically by stage `corr` from the table of every frame of the run; never set
`CORR_SHOTS` (an audit option) for a Final run and never copy a table from another run.

**Three Final submissions against the reserve of 800,000 cpu-minutes kept for the Final phase:**

| plan | what runs | allocated cpu-min |
|---|---|---|
| A — one Final run; submissions 2 and 3 are other post-stage settings on the same solve, generation and network outputs (e.g. `dec_e4s_c1nx`, then `int_b50c1nx`, then `syn_xa_p7li50`; same DIII-D file). A run directory belongs to one profile: assemble the second one from hard links to the first (`cp -al` of `shards/`, the merged `parts/`, `files_*`, `cfg_*`, `frames_*`, `scal_*`, `summary_*`, `post/packed_*`, `syn/pred`, `syn/models`) and run `STAGES="post validate"` with the other `PIPE` — the rehearsal of 14 October did exactly this | 1 complete submission + 2 × (post stages + MAST validator, about 800 + 1,450) | about 67,000 |
| B — three independent complete runs (e.g. a different inverse for one of them) | 3 × one complete submission | about 188,000 (about 163,000 with the 4-job solve) |
| C — plan B plus one complete re-run of everything after a fault | 4 × one complete submission | about 251,000 |
| peak booking of one run (the scheduler books cores × time limit of every RUNNING job against the project limit) | upper bound with the driver's limits: 32 × 72 × 22 (solve) + 32 × 72 × 18 (synthetic generation) + 288 × 70 min (DIII-D predict) ≈ 112,000, plus the DIII-D stream order 72 × 75 ≈ 5,400; through the dependencies the DIII-D predict starts only after the shard jobs have ended (in the dress rehearsal the 64 shard jobs and the DIII-D stream order ran at the same time: ≈ 98,000 booked) | **about 112,000 + 5,000 ≈ 117,000 (6.8 node-hours)**: keep at least this much headroom under any automatic guard that cancels jobs |

Even plan C uses less than a third of the reserve; the binding constraints are elsewhere: (i) the project limit counts used +
booked, so nothing else may run on the account on Final day; (ii) any automatic guard that cancels jobs above a usage threshold
set during the Development phase must be raised or stopped before the first Final job; (iii) GPU-hours are not a constraint
(< 0.3 GH200 GPU-hours per submission).

### B.8 Profile `dec_e4s_c1nx`: Final-day sequence, checks, cost, corpus statistics

`dec_e4s_c1nx` = the blend of `syn_xa_p7li50` → post stage `dec` (start-up frames re-decoded) → post stage `corr` (corrector setting
`c1nx` on the decoded maps) → pack. Its Development file (#17, md5 `006ed728…`) was reproduced as a whole file through this profile
(PROVENANCE.md section 3). Prerequisites in addition to section A: `src/.vendor_scorer` (`env/build_env.sh`), `weights/MANIFEST_CH2.sha256`
fetched (corrector model `mB50_tr.pkl` and the decoder's `support_source_d3d.parquet`; `tools/fetch_weights.sh` prints `WEIGHTS_OK`).

```bash
export FEC_WORK=/path/to/work/final; cd <package checkout>; P=dec_e4s_c1nx; MAST=mast_private_test
# after B.1 (inputs): MAST CPU stages, one job per shard, in parallel                                      [15 min / 12 min per shard]
PIPE=$P NSHARD=32 STAGES=init bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz          # once, before the shard jobs
for k in $(seq 0 31); do PIPE=$P NSHARD=32 SHARD=$k STAGES=solve  bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz; done
for k in $(seq 0 31); do PIPE=$P NSHARD=32 SHARD=$k STAGES=syngen bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz; done
PIPE=$P NSHARD=32 STAGES=synreal bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz                       # [15 s]
# one GPU: network training on the synthetic set + prediction of every frame                                 [4 min]
PIPE=$P NSHARD=32 STAGES=syntrain bash ch2/predict_mast.sh $FEC_WORK/in/mast final sorted $MAST.npz
# after B.2 (stream order): finish = merge, beta_N table, pack (p7li file), post synblend -> dec -> corr, stream order, local check
# ONE job, 72 cores, time limit 15 min                                                                       [about 9 min]
PIPE=$P NSHARD=32 STAGES="merge scal pack post validate" bash ch2/predict_mast.sh $FEC_WORK/in/mast final $FEC_WORK/order/mast/stream_order.txt $MAST.npz
# then B.6 (validator of $MAST.npz) and B.7 (publish the run-trained network model_mCxa32.pt) exactly as above
```

Checks of the decoder stage (finish-job log; public-test values for orientation; N = all Final frames):

| line | must hold | public test (69,075 frames) |
|---|---|---|
| `KTAB mast C4F:`, `KTAB mast mCxa32fg1:`, `KTAB mast mbl:` | each `N frames, 1205 shots`; `w contour fail` small | 69,075 / 1,206; 0.0000 |
| `DWEST … : N frames` | regime shares printed (start-up share expected 0.1–0.25) | start-up 0.170, near-DN 0.767 |
| `DSUP m: N frames` | support mean printed | 0.772 |
| `DWAPPLY … : N frames` | 13 target scales printed, all finite and > 0 | κ 0.167, V 1.084 |
| `DECRUN {…}` | `n_frames` = N; decoded + kept + skip + other_regime = N; `closure_max` < 1e-10 | 10,688 / 323 / 754 / 57,310; 7.4e-15 |
| `post dec -> <md5>` then `KTAB mast dec: N frames` and `KAPPLY … "n": N` | present, no `CORR_SHOTS selects` | — |
| `record.json` | `models`: `mB50_tr.pkl` 150620dd…, `support_source_d3d.parquet` dd08f384…; `constants`: `wD_L.json`, `wS_L.json`, `decoder_E4.json` of the frozen package | — |

**Corpus statistics recomputed on the Final inputs** (each post stage deletes its own directory and recomputes them from this run's maps
of the Final shots; no table of another run is read):

| statistic | stage / file of the run | over what |
|---|---|---|
| the network itself (trained on synthetic equilibria driven by the Final input columns) | `syngen`, `syntrain` / `syn/models/model_mCxa32.pt` | all Final shots (transductive, label-free) |
| target scales s_j of the decoder (standard deviation of the 13 per-frame targets) | `dec` / `dec/tg_m_stats.json` | all Final frames |
| pooled flux variance v_psi of the blend maps (normalises the decoder's flux term) | `dec` / `dec/run/cfg_dec_mast.json` | all Final frames |
| corrector feature centring (medians of the shot medians of the decoded maps' shape quantities) | `corr` / `corr/out/apply_corr_mast.json` (`ref`) | all Final shots (12 frames per shot) |
| corrector zero mean (mean of the predicted corrections, removed per output) | `corr` / `apply_corr_mast.json` (`corr_mean_cm`) | all valid Final frames |
| per shot, not across shots: frame regime (plasma current relative to the shot's own maximum), corrector shot medians, the 3-frame median of the packer | `dec`, `corr`, pack | each shot |
| per frame: support distance to the DIII-D training pool (whitening and thresholds come from the shipped DIII-D table) | `dec` / `dec/sup/sup_m.parquet` | each frame |

Fixed artefacts (never recomputed on the Final inputs): `ch2/constants/dec_e4s/wD_L.json` (DIII-D TRAIN pool), `wS_L.json` (our own synthetic
equilibria of the development phase), `decoder_E4.json`, the support source table, the corrector model, the calibration constants of the
network and of the inverse. Because the corrector's statistics are corpus-level, re-decoding the start-up frames also shifts the other
frames slightly (public test: about 0.3 cm at the plasma tips, l_i × 1.009) — a property of the recipe, not a setting.

**Option `dec_e4s_c1nx_a1w5`** (same maps; q95 by rule A1, β_N by rule B7w5): the same commands with `P=dec_e4s_c1nx_a1w5`. Its
finish job runs one more post stage, `a1w5` (about 165 s on 72 cores: two record tables of the corrected and of the decoded maps,
`OREC {… "frames": N, "fatal": 0 …}` twice, then `OAPPLY {… "n_shots": 1205 …}` and `post a1w5 -> <md5>`); the corrected file of the
stage before it is kept as `post/3_corr_mast_private_test.npz` (= what `dec_e4s_c1nx` alone would submit). The stage computes no
statistic across shots: A1 is a per-frame ratio of two contour integrals of our own maps, B7w5 a 5-frame median within each shot.
Time limit of the finish job 20 min; about 200 more allocated cpu-minutes.

Cost of one complete `dec_e4s_c1nx` submission: as `int_b50c1nx` in the budget table plus about 350 allocated cpu-minutes for the stage
`dec`, i.e. about 63,400 as rehearsed (about 54,400 with the 4-job solve); the decoder itself uses about 0.75 cpu-seconds per decoded
(start-up) frame and needs no GPU. A post-stage-only second submission on the same outputs (plan A, e.g. `int_b50c1nx` after `dec_e4s_c1nx`):
`STAGES="post validate"` in a hard-linked run directory, about 610 + 1,450 (validator) cpu-minutes.

### B.9 Dress rehearsal of 14–15 October (the Final procedure on the public test)

Section B was run once end to end as on Final day, with the frozen Final pair (`hc10x24` + `dec_e4s_c1nx`, the `DEFAULT` profiles):
a fresh checkout of the package (`git archive` of commit `bc2cb43ec5223a6d815d93afd528ac636ea3cd5c`; the two Python environments and
the starter kit were links to an earlier clean room built from the same lock files, the scorer overlay was installed from its lock),
weights placed by `tools/fetch_weights.sh` from local copies of the weight archives (sha256-checked; the public Hub route of step A.6
was not exercised), inputs DOWNLOADED from the Hub by B.1 (dataset revision `1e280905…`, 874 + 1,206 files, 7.19 + 3.24 GB),
everything else computed in the run: stream orders, inverse solve, synthetic set, the network trained afresh on it, blend, start-up
decoder and corrector with every corpus statistic recomputed from these inputs. One call:
`REPO=Sophelio/fusion-equilibrium-challenge D3D=diii_d_public_test MAST=mast_public_test DEVICE=gpu bash tools/final_slurm.sh submit`
(no read token; the 1-GPU probe started after 1 s).

| step | jobs × shape | wall (longest job) | allocated cpu-min | node-hours |
|---|---|---|---|---|
| B.1 download | 1 × 72 cores | 356 s | 427 | 0.025 |
| B.2 stream order DIII-D / MAST | 2 × 72 cores, one after the other | 2,469 s / 2,179 s | 2,963 / 2,615 | 0.171 / 0.151 |
| B.3 `init` + `synreal` | 2 × 72 cores | 9 s / 17 s | 31 | 0.002 |
| B.3 inverse solve | 32 × 72 cores | mean 586 s, max 869 s | 22,507 | 1.302 |
| B.3 synthetic generation | 32 × 72 cores | mean 534 s, max 728 s | 20,495 | 1.186 |
| B.4 network training + prediction | 72 cores + 1 GPU | 258 s (training 162 s) | 310 | 0.018 |
| B.5 DIII-D predict + gate | 288 cores + 1 GPU, 400 GB | 2,281 s | 10,949 | 0.634 |
| B.5 MAST finish | 72 cores | 548 s | 658 | 0.038 |
| B.6 validators DIII-D / MAST | 2 × 72 cores, one after the other | 2,021 s / 1,959 s | 2,425 / 2,351 | 0.140 / 0.136 |
| **total** | **74 jobs** | **2 h 32 min from submission to both files validated** (2 h 26 min after the download) | **65,730** | **3.80** |

Results (`tools/final_slurm.sh check`: 24 / 24 PASS):
- `diii_d_public_test.npz` md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c` = the Development file of `hc10x24`; log lines `heads v2.2 c10 (24 seeds)`,
  `HEADS_CHECK OK`, `ALL_DONE`; gate 8 PASS / 2 REVIEW (G7 sign, G9 fallback statistics, as on every public-test run) / 0 FAIL; G6 PASS;
  organisers' validator OK.
- `mast_public_test.npz` md5 `006ed7289f53e51c9602e8326b75c3c4` = the Development file of `dec_e4s_c1nx` (#17), with the network trained
  in the run: its 129 weight arrays equal those of the Development network (the file's sha256 differs only through stored metadata);
  every intermediate equals its development counterpart (solve outputs, β_N table, network inputs, synthetic set apart from its
  wall-clock column, network predictions, blend `11d94769…`, functional / estimator / support / target tables, decoder record, decoded
  file `ef440229…`, corrector table and statistics, corrected maps); organisers' validator OK.

**Final-day checkout (package freeze of 15 October).** The freeze commit is the direct child of commit
`c0e673388362fea076e550c649b47d2994cbb159` and changes documentation only (this paragraph and one row of PROVENANCE.md section 3). Check
the checkout before the first job: in a git clone `git log -1 --format=%P HEAD` prints that SHA and `git diff --stat c0e6733 HEAD`
lists only `RUNBOOK_FINAL.md` and `PROVENANCE.md`; for a directory extracted from a `git archive` tarball, `gunzip -c <tarball> |
git get-tar-commit-id` prints the freeze commit's SHA. That SHA is recorded outside the package (a commit cannot name itself); step
B.7 adds the run-trained network in one further commit, whose SHA goes into the Code URL.

Changes after the rehearsed commit `bc2cb43` (`git diff --stat bc2cb43 c0e6733`): the MAST local format check follows the stream order
(`validate_local.py --order`, passed by the MAST validate stage and by `check`); `tools/fetch_weights.sh` fetches the weights of the
selected profiles by default; `tools/final_slurm.sh` checks every weight file against its manifest before it submits anything
(`PREFLIGHT`), compares every sha256 of `record.json` with the checkout, refuses to re-submit a started `ch1` / `finish` step without
`REDO=1`, and carries the `status` / `check` / time-limit changes made after the rehearsal; the two predict entrypoints print the
Development md5 only for public-test output names; the geometry fetch script uses a fixed list of reference shots; documentation.
None of this changes a number in either file. The file tree of commit `c0e6733` (git tree `4c9fa0b7b5fc…`) was re-verified on 15 October
from a fresh extraction in a clean room (environments linked from an earlier clean room built from the same lock files; weights placed
by `tools/fetch_weights.sh`):
- DIII-D through `ch1/predict.sh` (288 cores + 1 GPU, this rehearsal's inputs and stream order): md5 `0c9beef87ee0b8ac63c1f628f7ccdf2c`;
- MAST through `ch2/predict_mast.sh` (pack, blend, start-up decoder, corrector, local check) on the solve and network outputs of the
  Development run (this rehearsal recomputed them with identical arrays): md5 `006ed7289f53e51c9602e8326b75c3c4`; `check` rows "MAST local format
  check (stream order)" and "record.json = frozen package" (18 sha256) PASS;
- the same MAST path under the private-test output name with a permuted stream order: the sorted-order file md5 `006ed728…`, the final
  file its exact key permutation, the local check OK in stream order (and FAIL without the order file);
- unit tests of the package 41 / 41 PASS (bash 5, Python 3.11); `DRYRUN=1 … submit` with the Final defaults: `PREFLIGHT OK`, 74 jobs.

Not rehearsed: a read token, the Final repository and its configuration names (the driver takes them as parameters; the download
tool falls back to the dataset card), step B.7 (which is why its shipped-weights check is mandatory on Final day);
`ch2/train_mast_net.sh` itself (added after the Final checkout; run only with DRYRUN=1 and against a stub; its calls are those of
the rehearsed final_slurm.sh jobs).

### B.10 Third submission (F3): DIII-D profile `xhc`, MAST file of F1

DIII-D only (the MAST file is F1's, byte-identical). From a fresh checkout of commit `453d04c` (branch `f3-x`; every file the
`hc10x24` / `dec_e4s_c1nx` runs execute is byte-identical to `82e50ff`) in a NEW work directory: inputs and the 73 weight files of
`MANIFEST.sha256` + `MANIFEST_CH1_HC10X24.sha256` as per-file read-only links to the first run's copies, the 32 files of
`MANIFEST_CH1_XHC.sha256` placed from the xhc archive (sha256-verified), the stream order derived again from the Hub:

```bash
export FEC_WORK=<new work dir> HF_TOKEN_FILE=<token file> CH1_PROFILE=xhc PIPE=dec_e4s_c1nx
export REPO=ganatma/fusion-equilibrium-challenge-final D3D=diii_d_private_test MAST=mast_private_test RUN=final_xhc DEVICE=gpu
STEPS="order_d3d ch1 val_d3d" CPUS_CH1=72 MEM_CH1=100G TIME_CH1=03:00:00 bash tools/final_slurm.sh submit   # PREFLIGHT: 49 + 24 + 32 files
# then: stream_order.txt = the first run's (cmp); the organisers' validator at 4b6af06: validate_submission.py <file> --config diii_d_private_test --max-shots 0
```

Checks: `HEADS_CHECK OK: profile xhc`, `G6 zero derivation failures: PASS`, gate 0 FAIL, `ALL_DONE`; both validators OK (879 shots);
keys, shapes, dtypes and byte count equal to the F1 DIII-D file; q95 / β_N bit-identical to it (same SC members); the flux maps
differ by the fine-tune (per-frame max |Δψ| / span, `xhc` against `hc10x24`: Final file against the F1 file median 1.35e-3,
99th percentile 7.94e-3, maximum 0.0873; public test through the package path median 1.41e-3, 99th percentile 8.19e-3, maximum
0.151). Result (job 7554839): md5
`20753614af6adb28b249a250baf803c7`, sha256 `9c91bdaa41f1391b26d4636cc2e098c8d6c3cc7f358e45da0c7dc3affe63eab1`, 879 shots, both
validators OK; a re-run of the same inference gave a bit-identical file (0 differing elements).

## C. Submission

1. `src/starter/push_predictions.py` (the organisers' pointer route) with the two files in one directory named exactly
   `diii_d_private_test.npz` and `mast_private_test.npz`: private predictions repository + `submission_pointer.zip`. The pointer
   zip holds a read token: never print it, never commit it.
2. **[decision]** Upload order (3 submissions): #1 the frozen package output (`<C>` + `<P>`). #2 / #3 only for a pre-decided
   alternative MAST profile or a platform fault (a slot lost to a platform fault is returned on request). The Award #2 denominator
   is the team's best DIII-D composite across its scored Final submissions, so the DIII-D file is the same file in every
   submission. Each submission needs its own package commit if its MAST profile or its run-trained network differs.
3. Fact sheet Code URL: the public repository at the pinned 40-character commit (or the archive URL + sha256). For a `syn_*`
   profile this is the commit of step B.7, which contains the network of that submission.
4. Archive locally (not in the package): both npz md5 / sha256, `record.json`, gate and validator logs, the original and the
   published network sha256, submission IDs, times.

## D. After the close

- Top-five check by the organisers (forum post 2872, 18 October): (1) retraining from the training entrypoints, scored on the
  same fold, must give a composite S within 0.01 of the submitted S on each machine ranked (MAST network:
  `ch2/train_mast_net.sh`); (2) the shipped weights through the inference entrypoint must reproduce the submitted S within 1e-4
  (MAST: `SYN_MODEL_FILE=<published network>` with `ch2/predict_mast.sh`, README.md "MAST inference with the shipped network";
  the synthetic set is still generated; pass the stream-order file). Failing either keeps the entry on the leaderboard but not
  award-eligible. Compare with PROVENANCE.md section 3. What to expect of a rebuild elsewhere: the DIII-D file and an inverse
  MAST file as described there; for a `syn_*` file, predictions from the
  PUBLISHED network are reproduced like an inverse file, whereas retraining the network on other hardware gives a different
  network (not measured across GPU types; on the reference hardware the retrained network is bit-identical).
- Top three per award: OSI licence (MIT, done) and a 1–2 page methods report (METHODS.md) before prizes are paid.
