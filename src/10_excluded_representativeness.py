# -*- coding: utf-8 -*-
"""Representativeness of the excluded gauges (Table S16).

Compares the 307 analysable stations with the gauges bound outside the
foul/combined/storm sub-networks, and recomputes the main slope under
post-stratification to the excluded stations' upstream-scale distribution.

Inputs (in ../derived):
    p1_h3_master.csv           station-level analysis table (up_size, se_tpk_max, log_up)
    p5_excluded_compare.csv    excluded gauges with a bound node (st, system, up_size, bind_m)

Output: printed table; ../results/S16_excluded.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

T = os.environ.get("SEWER_TIMING_DATA",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "results")

D = pd.read_csv(os.path.join(T, "p1_h3_master.csv"))
E = pd.read_csv(os.path.join(T, "p5_excluded_compare.csv"))

M = D[D["se_tpk_max"].notna()].reset_index(drop=True)
keep = M["up_size"].dropna().values
excl = E[E["grp"] == "excluded"]["up_size"].dropna().values


def wls(x, y, w):
    A = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    b = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    return b[1]


print("Panel A  station groups")
print(f"  retained                n={len(keep):4d}  median={np.median(keep):5.1f}  "
      f"IQR={np.percentile(keep, 25):.1f}-{np.percentile(keep, 75):.1f}")
print(f"  excluded, bound         n={len(excl):4d}  median={np.median(excl):5.1f}  "
      f"IQR={np.percentile(excl, 25):.1f}-{np.percentile(excl, 75):.1f}")
print(f"  Mann-Whitney p = {stats.mannwhitneyu(excl, keep, alternative='two-sided').pvalue:.4f}")

x, y = M["log_up"].values, M["se_tpk_max"].values
b_raw = wls(x, y, np.ones_like(x))
print(f"\nPanel B  slope, unweighted: {b_raw:.3f}")

rows = []
schemes = {
    "quintiles of the excluded distribution (primary)": list(np.unique(np.percentile(excl, [20, 40, 60, 80]))),
    "fixed cuts 1,2,4,8,16,64": [1, 2, 4, 8, 16, 64],
    "fixed cuts 1,2,4,8,32": [1, 2, 4, 8, 32],
    "fixed cuts 1,2,4,8,16,64,256": [1, 2, 4, 8, 16, 64, 256],
    "fixed cuts 1,2,4,8": [1, 2, 4, 8],
}
for label, cuts in schemes.items():
    cuts = np.array(cuts, dtype=float)
    kb, ke = np.digitize(keep, cuts), np.digitize(excl, cuts)
    pr = pd.Series(kb).value_counts(normalize=True)
    pe = pd.Series(ke).value_counts(normalize=True)
    w = np.array([pe.get(k, 0.0) / pr.get(k, 1e-9) for k in kb])
    b = wls(x, y, w)
    rows.append({"scheme": label, "beta": round(b, 3),
                 "change_pct": round(100 * (b - b_raw) / abs(b_raw), 1),
                 "w_min": round(w.min(), 2), "w_max": round(w.max(), 2)})
    print(f"  {label:48s} beta={b:+.3f}  ({(b-b_raw)/abs(b_raw)*100:+.1f}%)  w in [{w.min():.2f},{w.max():.2f}]")

os.makedirs(R, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(R, "S16_excluded.csv"), index=False, encoding="utf-8-sig")
print("\nwritten: results/S16_excluded.csv")
