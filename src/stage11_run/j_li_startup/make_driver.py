#!/usr/bin/env python3
"""
Stage 11J / li_startup — builds run_inv7l.py from g_restrict/run_inv7.py by ASSERTED string patches (each pattern must occur
exactly once), so that the diff against the restrict-track driver is exactly this list.  With every flag at its default
(li_model = None, gap_w = 0, fb_gate = 0, no --shard) the patched driver executes the statements of run_inv7.py unchanged.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "g_restrict" / "run_inv7.py").read_text()
P = []

P.append(('''Stage 11G / Ch2 TRACK restrict — run_inv7.py =''',
          '''Stage 11J / Ch2 TRACK li_startup — run_inv7l.py = g_restrict/run_inv7.py (statements unchanged on the default path) plus
  * li_model      json of the DIII-D-trained ridge model (li_fit.py): li predicted from dimensionless ramp-history features
                  + Thomson exponents enters the LM objective as a Gaussian pseudo-observation of the solution's li
                  (li_sig_scale, li_b1_var / li_b2_var relax the static FF' prior so the datum replaces it), li_core.py
  * gap_w         sd [m] of a ONE-SIDED wall-clearance datum inside the LM objective (start-up frames; no field push)
  * fb_gate       audit fix F1 (g_audit/run_inv6a.py): a G1 fallback is shipped only if it passes the inverse's own test
  * --shard i/n   solve only the shots with index % n == i (parts only); a later call without --shard collects
Original header of run_inv7.py:
Stage 11G / Ch2 TRACK restrict — run_inv7.py ='''))

P.append(('''sys.path.insert(0, str(HERE))                             # this track's own modules first
''', '''sys.path.insert(0, str(HERE.parent / "g_restrict"))       # inv7_core (unchanged restrict-track module)
sys.path.insert(0, str(HERE))                             # this track's own modules first
'''))

P.append(('''import inv7_core as K7  # noqa: E402
''', '''import inv7_core as K7  # noqa: E402
import li_core as LI  # noqa: E402

_LIM = {}


def li_model_of(path):
    if path not in _LIM:
        _LIM[path] = LI.LiModel(path)
    return _LIM[path]
'''))

P.append(('''    cfg = {**K.INV_DEFAULT, **INV6_DEFAULT, **INV7_DEFAULT, **S.cfg}
''', '''    cfg = {**K.INV_DEFAULT, **INV6_DEFAULT, **INV7_DEFAULT, **LI.LI_DEFAULT, **S.cfg}
'''))

P.append(('''    th = D.thomson_frames(row, machine, t, cfg)
    lab = D.labels_of(row) if machine == "d3d" else {}
''', '''    th = D.thomson_frames(row, machine, t, cfg)
    li_shot, li_cdef = None, 1.0
    if cfg.get("li_model") and mode != "labelfit":
        li_shot = LI.ShotLi(li_model_of(cfg["li_model"]), row, machine, t, th, cfg, fvac=inp["fvac"])
        li_cdef = float(cfg.get("li_cdef") or li_shot.model.cdef)
    lab = D.labels_of(row) if machine == "d3d" else {}
'''))

P.append(('''            prof_k, edges, info = pf
            edges = edge_filter(edges)
''', '''            prof_k, edges, info = pf
            info["_prof"] = prof_k
            edges = edge_filter(edges)
'''))

P.append(('''        if machine == "mast":
            pk = th["prof"][k]
            iso = K.mast_iso_obs(pk, levels, chord[1]) if pk is not None else []
''', '''        if machine == "mast":
            pk = th["prof"][k]
            info["_prof"] = pk
            iso = K.mast_iso_obs(pk, levels, chord[1]) if pk is not None else []
'''))

P.append(('''        oinfo = dict(oinfo)
        iso_n = oinfo.pop("iso_n", [])
''', '''        oinfo = dict(oinfo)
        iso_n = oinfo.pop("iso_n", [])
        lprof = oinfo.pop("_prof", None)
'''))

P.append(('''        else:
            rfn = None                                   # INV_P5 path (inv_core.residuals)
        sol, info = inv_fn(fm, x0, zt0, warm, iso, edges, 0.0 if mode == "labelfit" else z_eta, a_pub, zt_max,
                           params=params if has_data else [], resid_fn=rfn, x_prior=x_prior)
        info["n_dec"] = n_dec
''', '''        else:
            rfn = None                                   # INV_P5 path (inv_core.residuals)
        # ---- li_startup: li pseudo-observation (prior on the FF' shape, in li) + one-sided wall clearance, inside the LM
        lip, li_t, li_s = None, None, None
        if li_shot is not None:
            lip = li_shot.frame(k, lprof)
            if lip["pred"] is not None:
                li_t = lip["pred"]["li"] + float(cfg.get("li_offset", 0.0))
                li_s = max(float(np.hypot(cfg["li_sig_scale"] * lip["pred"]["sig"], cfg["li_sig_add"])), cfg["li_sig_min"])
        if li_shot is None and cfg.get("prior_b_var", 1.0) != 1.0 and mode != "labelfit":       # diagnostic: weak static FF' prior
            xm_, xv_ = x_prior if x_prior is not None else ({}, {})
            x_prior = (dict(xm_), {**xv_, "b1": xv_.get("b1", 1.0) * cfg["prior_b_var"], "b2": xv_.get("b2", 1.0) * cfg["prior_b_var"]})
        gap_w = float(cfg.get("gap_w", 0.0)) if mode != "labelfit" else 0.0
        if li_t is not None or gap_w > 0:
            base_rfn = rfn

            def rfn(s_, base_rfn=base_rfn, iso=iso, edges=edges, li_t=li_t, li_s=li_s, gap_w=gap_w):
                if base_rfn is not None:
                    r_, h_ = base_rfn(s_)
                else:
                    r_, h_, _ = K.residuals(s_, iso, edges, cfg)
                er, eh = LI.extra_residuals(S, s_, Ip[k], li_t, li_s, li_cdef, gap_w, cfg["gap_min_lm"], cfg["li_huber"],
                                            datum=li_shot.model.datum if li_shot is not None else "li")
                return np.append(r_, er), np.append(h_, eh)
            if li_t is not None:
                has_data = True
                if cfg["li_b1_var"] != 1.0 or cfg["li_b2_var"] != 1.0:
                    xm_, xv_ = x_prior if x_prior is not None else ({}, {})
                    x_prior = (dict(xm_), {**xv_, "b1": xv_.get("b1", 1.0) * cfg["li_b1_var"], "b2": xv_.get("b2", 1.0) * cfg["li_b2_var"]})
        sol, info = inv_fn(fm, x0, zt0, warm, iso, edges, 0.0 if mode == "labelfit" else z_eta, a_pub, zt_max,
                           params=params if has_data else [], resid_fn=rfn, x_prior=x_prior)
        info["n_dec"] = n_dec
        if lip is not None:
            info["li_pred"] = lip
'''))

P.append(('''            r1b, _ = g1_start(k, edges)
            if r1b is not None:
                r1b["g1"] = True
                sol = r1b
''', '''            r1b, _ = g1_start(k, edges)
            if r1b is not None and cfg.get("fb_gate"):         # audit fix F1 (g_audit/run_inv6a.py, same three constants)
                lo_, hi_ = cfg["axis_range"][machine]
                dts_ = K.edge_deltas(r1b, [o_ for o_ in edges if o_["name"] == "out" and o_["kind"] == "eq"])
                if int(cfg["fb_gate"]) >= 2 or not (lo_ <= r1b["b"]["axis"][0] <= hi_) or bool(r1b.get("clip")) \\
                        or any(abs(v_) > cfg["closure_tol"] for v_ in dts_.values()):
                    info["err"] = (info.get("err", "") + " -> G1 fallback REJECTED (fails the validity gate)").strip()
                    info["g1_rejected"] = True
                    r1b = None
            if r1b is not None:
                r1b["g1"] = True
                sol = r1b
'''))

P.append(('''        rec["n_dec"] = float(info.get("n_dec", np.nan))
''', '''        rec["n_dec"] = float(info.get("n_dec", np.nan))
        rec["g1_rejected"] = bool(info.get("g1_rejected", False))
        lp_ = info.get("li_pred")
        if lp_ is not None:
            rec.update({f"lf_{n_}": float(v_) for n_, v_ in lp_["feat"].items()})
            if lp_["pred"] is not None:
                pp_ = lp_["pred"]
                rec.update({"li_t": pp_["li"], "li_raw": pp_["li_raw"], "li_sig": pp_["sig"], "li_d": pp_["d"], "li_insup": float(pp_["insup"]),
                            "li_nclip": pp_["nclip"], "li_full": float(pp_["model"] == "full")})
        if sol is not None and sol.get("ok"):
            try:
                dw_ = LI.wall_gap(S, sol, cfg["gap_min_lm"])
                rec["gap_dw"] = float(dw_) if dw_ is not None else np.nan
            except Exception:  # noqa: BLE001
                rec["gap_dw"] = np.nan
            rec["gs_lic"] = LI.solver_lic(S, sol)
'''))

P.append(('''    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
''', '''    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--shard", default=None, help="i/n: solve only the shots with index %% n == i and write parts only")
    args = ap.parse_args()
'''))

P.append(('''    json.dump({**K.INV_DEFAULT, **C.DEFAULT_CFG, **INV6_DEFAULT, **INV7_DEFAULT, **{k_: v_ for k_, v_ in cfg.items() if k_ != "anchors"},
''', '''    json.dump({**K.INV_DEFAULT, **C.DEFAULT_CFG, **INV6_DEFAULT, **INV7_DEFAULT, **LI.LI_DEFAULT, **{k_: v_ for k_, v_ in cfg.items() if k_ != "anchors"},
'''))

P.append(('''    tasks = [(args.machine, str(p), i, cfg, args.wall, str(out), args.tag, args.frames_per_shot, args.mode) for i, p in items]
    t0 = time.time()
''', '''    tasks = [(args.machine, str(p), i, cfg, args.wall, str(out), args.tag, args.frames_per_shot, args.mode) for i, p in items]
    if args.shard:
        si_, sn_ = (int(x) for x in args.shard.split("/"))
        tasks = [tk for tk in tasks if tk[2] % sn_ == si_]
        tasks.sort(key=lambda tk: -os.path.getsize(tk[1]))    # longest shots first (pool packing; results are per-shot deterministic)
    t0 = time.time()
'''))

P.append(('''    import pandas as pd
    df = pd.DataFrame(recs)
    for c in df.columns:
''', '''    if args.shard:
        log(f"shard {args.shard}: {len(tasks)} shots, {len(recs)} frames, parts only ({time.time() - t0:.0f}s)")
        return
    import pandas as pd
    df = pd.DataFrame(recs)
    for c in df.columns:
'''))

P.append(('''        cfg[k] = json.loads(v)
''', '''        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:                           # plain string value (paths)
            cfg[k] = v
'''))

P.append(('''    Ip = np.abs(inp["ip_A"])
''', '''    Ip = np.abs(inp["ip_A"]) * (1.0 + float(cfg.get("twin_eps", 0.0)))      # replica knob (audit): last-digit perturbation of Ip
'''))

P.append(('''    log(f"INV7 {args.machine} mode=''', '''    log(f"INV7L {args.machine} mode='''))

out = src
for a, b in P:
    assert out.count(a) == 1, f"pattern occurs {out.count(a)} times:\\n{a[:120]}"
    out = out.replace(a, b)
(HERE / "run_inv7l.py").write_text(out)
print(f"run_inv7l.py written ({len(P)} patches, {len(out)} bytes)")
