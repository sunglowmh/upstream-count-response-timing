# -*- coding: utf-8 -*-
"""
R-04 补强：标定曲线在 20 组地理分块上的稳健性
==============================================
k ∈ {3,4,5,6} × seed ∈ {0..4}，共 20 组，与 Fig. 4 Panel B / Table 2 同一网格。
每组报 m=0 / 10 / 20 的 zero-label pooled Spearman 与平均折内 RMSE。
"""
import os
import numpy as np, pandas as pd, io, sys
from scipy.cluster.vq import kmeans2
from scipy import stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
n = len(m)
y = m["se_tpk_max"].to_numpy(float)
log_up = m["log_up"].to_numpy(float)
Xs = np.column_stack([m["x_std"].to_numpy(float), m["y_std"].to_numpy(float)])
X1 = np.column_stack([np.ones(n), log_up])

MS = [0, 5, 10, 20]
REPS = 100
MIN_EV = 8

rows = []
for k in [3, 4, 5, 6]:
    for seed in range(5):
        lab = kmeans2(Xs, k, seed=seed, minit="points")[1]
        folds = []
        for j in range(k):
            te = np.where(lab == j)[0]
            tr = lab != j
            if len(te) < MIN_EV + 2 or tr.sum() < 5:
                continue
            b, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None)
            folds.append(dict(te=te, slope=b[1], b0=b[0], n=len(te)))
        if len(folds) < k:
            continue
        rng = np.random.default_rng(20261105 + 101 * k + seed)
        rec = dict(k=k, seed=seed, n_folds=len(folds))
        for mm in MS:
            rm, sp = [], []
            for _ in range(REPS):
                yt, yp, rf = [], [], []
                for d in folds:
                    te, slope, cnt = d["te"], d["slope"], d["n"]
                    kk = min(mm, max(0, cnt - MIN_EV))
                    if kk > 0:
                        cal = rng.choice(te, size=kk, replace=False)
                        ev = np.setdiff1d(te, cal)
                        ic = y[cal].mean() - slope * log_up[cal].mean()
                    else:
                        ev, ic = te, d["b0"]
                    yh = ic + slope * log_up[ev]
                    rf.append(np.sqrt(((y[ev] - yh) ** 2).mean()))
                    yt.extend(y[ev]); yp.extend(yh)
                rm.append(np.mean(rf))
                sp.append(stats.spearmanr(yt, yp).correlation)
            rec[f"rmse_{mm}"] = float(np.mean(rm))
            rec[f"rho_{mm}"] = float(np.mean(sp))
        rows.append(rec)

d = pd.DataFrame(rows)
d.to_csv(f"{T}/p5_calib_grid.csv", index=False, encoding="utf-8-sig")
print(f"可用分块 {len(d)} / 20")
print()
hdr = f"{'m':>4} {'pooled rho: min / median / max':>36} {'mean':>7}   {'foldRMSE: min / median / max':>34}"
print(hdr); print("-" * len(hdr))
for mm in MS:
    s, r = d[f"rho_{mm}"], d[f"rmse_{mm}"]
    print(f"{mm:>4} {s.min():.3f} / {s.median():.3f} / {s.max():.3f}".ljust(42)
          + f"{s.mean():.3f}   " + f"{r.min():.2f} / {r.median():.2f} / {r.max():.2f}")
print()
for mm in MS[1:]:
    d_rho = d[f"rho_{mm}"] - d["rho_0"]
    d_rm = d["rmse_0"] - d[f"rmse_{mm}"]
    print(f"m=0 → {mm:>2}: Δρ 中位 {d_rho.median():+.3f} (范围 {d_rho.min():+.3f} ~ {d_rho.max():+.3f}), "
          f"ΔRMSE 中位 {d_rm.median():+.2f} h (范围 {d_rm.min():+.2f} ~ {d_rm.max():+.2f})")
print()
print(f"Δρ>0 的分块数: " + ", ".join(f"m={mm}: {(d[f'rho_{mm}']>d['rho_0']).sum()}/{len(d)}" for mm in MS[1:]))
