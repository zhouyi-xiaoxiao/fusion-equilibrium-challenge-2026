#!/usr/bin/env python3
"""Put the rows of a per-shot table (parquet with a `file` column) into a recorded shot order.

  reorder_table.py <table.parquet> <shot order .txt> [<out.parquet>]        (default: rewrite in place)

The table builders write their rows in the completion order of a worker pool, which differs from run to run. A least-squares fit
sums over the rows in table order, so its coefficients depend on that order at the 1e-9 relative level. The shot order of the
development tables is shipped (splits/*_roworder.txt); with it restored, the fit returns the shipped model bit for bit. Rows of one
shot stay in the order the builder wrote them (stable sort); the set of shots must equal the list.
"""
import sys

import numpy as np
import pandas as pd

tab, lst = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else tab
order = [x for x in open(lst).read().split("\n") if x]
df = pd.read_parquet(tab)
rank = {n: i for i, n in enumerate(order)}
assert len(rank) == len(order), "duplicate names in the order list"
assert set(df["file"].unique()) == set(order), "the shots of the table differ from the order list"
idx = np.argsort(df["file"].map(rank).to_numpy(), kind="stable")
df.iloc[idx].reset_index(drop=True).to_parquet(out)
print(f"REORDER {len(df)} rows of {len(order)} shots -> {out}")
