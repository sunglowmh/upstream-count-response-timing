# -*- coding: utf-8 -*-
"""
R-04 本地标定样本量曲线（评审意见 P0，最高价值单项）
=====================================================
问题（DA-C1）
  正文 §3.4 的 "local intercept recalibration" 只用「整个留出子区」标定截距
  （极端情形），而 §5 又把「需要几个本地观测」列为设计无法回答的问题。
  两者之间缺一条 m 曲线。

本脚本
  对 k=5 地理分块（seed=0，与 Fig. 4 / 表 3 Panel B 同一分块）：
    每折在训练站上估斜率 slope（固定），在留出子区内随机抽 m 个站估截距
        intercept = mean(y_cal) - slope * mean(x_cal)
    其余留出站评估（校准站不入评估）
  m ∈ {0, 5, 10, 20, 40, all}
  重复 200 次重抽，报均值与 95% 区间。

两个口径
  (a) within-subregion RMSE（工程口径：小时）
  (b) zero-label pooled Spearman（部署口径：跨折汇总，不使用任何测试标签）
      —— 这是 §2.5 承认 "evaluation statistic only" 的那个量的可部署版本：
         校准用 m 个站的标签，评估时不需任何标签。

自检
  m = 0 的 RMSE 应 ≈ 9.06 h，m = all 应 ≈ 8.45 h（复刻正文 9.1→8.5）。
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy import stats
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
n = len(m)
y = m["se_tpk_max"].to_numpy(float)
log_up = m["log_up"].to_numpy(float)
Xs = np.column_stack([m["x_std"].to_numpy(float), m["y_std"].to_numpy(float)])
X1 = np.column_stack([np.ones(n), log_up])

lab = kmeans2(Xs, 5, seed=0, minit="points")[1]
print(f"n = {n}   子区规模 = {np.bincount(lab).tolist()}")

# 每折斜率（固定，与 m 无关）
folds = []
for j in range(5):
    te = lab == j
    tr = ~te
    b, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None)
    folds.append(dict(j=j, te=np.where(te)[0], slope=b[1], b0=b[0],
                      n_te=int(te.sum())))
print("每折: " + "  ".join(f"f{d['j']}: n={d['n_te']}, slope={d['slope']:+.3f}" for d in folds))

MS = [0, 5, 10, 20, 40, None]
REPS = 200
MIN_EV = 8          # 每折至少保留 8 个评估站（最小子区仅 20 站）
rng = np.random.default_rng(20261105)

res = {mm: dict(rmse=[], pooled=[], spear=[]) for mm in MS}
for mm in MS:
    for rep in range(REPS):
        yt, yp = [], []           # 校准后汇总（部署口径）
        rmse_folds = []
        for d in folds:
            te, slope, n_te = d["te"], d["slope"], d["n_te"]
            if mm is None:
                # 正文口径：用整个留出子区估截距，并在同一子区上评估
                k = n_te
                cal = te
                ev = te
            else:
                k = min(mm, max(0, n_te - MIN_EV))
                if k > 0:
                    cal = rng.choice(te, size=k, replace=False)
                    ev = np.setdiff1d(te, cal)
                else:
                    cal = np.array([], dtype=int)
                    ev = te
            if k > 0:
                ic = y[cal].mean() - slope * log_up[cal].mean()
            else:
                ic = d["b0"]
            yhat = ic + slope * log_up[ev]
            rmse_folds.append(float(np.sqrt(((y[ev] - yhat) ** 2).mean())))
            yt.extend(y[ev]); yp.extend(yhat)
        yt = np.asarray(yt); yp = np.asarray(yp)
        res[mm]["rmse"].append(float(np.mean(rmse_folds)))
        res[mm]["pooled"].append(float(np.sqrt(((yt - yp) ** 2).mean())))
        res[mm]["spear"].append(float(stats.spearmanr(yt, yp).correlation))

print()
print(f"{'m':>6} {'foldRMSE':>9} {'95% CI':>15} {'poolRMSE':>9} {'pooled rho':>11} {'95% CI':>15}")
print("-" * 74)
rows = []
for mm in MS:
    r = np.array(res[mm]["rmse"]); p_ = np.array(res[mm]["pooled"]); s = np.array(res[mm]["spear"])
    lab_m = "all" if mm is None else str(mm)
    print(f"{lab_m:>6} {r.mean():9.2f} "
          f"{'[' + format(np.percentile(r,2.5),'.2f') + ',' + format(np.percentile(r,97.5),'.2f') + ']':>15} "
          f"{p_.mean():9.2f} {s.mean():11.3f} "
          f"{'[' + format(np.percentile(s,2.5),'.3f') + ',' + format(np.percentile(s,97.5),'.3f') + ']':>15}")
    rows.append(dict(m=("all" if mm is None else mm), rmse_mean=r.mean(),
                     rmse_lo=np.percentile(r, 2.5), rmse_hi=np.percentile(r, 97.5),
                     pooled_mean=p_.mean(),
                     spear_mean=s.mean(), spear_lo=np.percentile(s, 2.5),
                     spear_hi=np.percentile(s, 97.5), n_reps=REPS))

out = pd.DataFrame(rows)
out.to_csv(f"{T}/p5_calib_curve.csv", index=False, encoding="utf-8-sig")
print(f"\n已保存 {T}/p5_calib_curve.csv")

# 自检
r0 = out.loc[out.m == 0, "rmse_mean"].iloc[0]
ra = out.loc[out.m == "all", "rmse_mean"].iloc[0]
print(f"自检：m=0 RMSE {r0:.2f} (正文 9.1) ; m=all RMSE {ra:.2f} (正文 8.5)")

# 阈值判定：哪一档之后改善进入平台
print()
for i in range(1, len(out)):
    d_r = out.rmse_mean.iloc[i - 1] - out.rmse_mean.iloc[i]
    d_s = out.spear_mean.iloc[i] - out.spear_mean.iloc[i - 1]
    print(f"  {str(out.m.iloc[i-1]):>4} → {str(out.m.iloc[i]):>4} : ΔRMSE = {d_r:+.2f} h, Δρ = {d_s:+.3f}")
