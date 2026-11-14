"""Feature names of syn_common.ts_feats without importing the solver stack (used by syn_predict --drop-feat)."""
LEVELS = (0.2, 0.35, 0.5, 0.65, 0.8)
FEAT_NAMES = (["out_eq", "out_eq_x", "out_ge", "out_ge_x", "in_eq", "in_eq_x", "in_ge", "in_ge_x"]
              + [f"{s}{f:g}{k}" for f in LEVELS for s in ("lo", "hi") for k in ("_has", "_x")]
              + ["prof_has", "tepk_x", "sym_has", "sym_x", "log_beta_pe"])
