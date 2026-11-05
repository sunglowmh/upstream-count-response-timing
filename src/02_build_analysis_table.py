# -*- coding: utf-8 -*-
"""P1-H3b：逐水力指标条件增量 + 总量 vs 均值机制分解。

补充 p1_h3 的结论，回答两个问题：
  Q1: up_size 相对「每一个」水力指标单独（非三者联合）是否仍有增量？
  Q2: up_size 的预测力来自「总量 Στ」还是「均值 Στ/N」？—— 这是"为什么有效"的机制钥匙。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
m["st"] = m["st"].astype(str).str.strip()

BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
ycol = "se_tpk_max"   # 主响应 global max


def ols_r2(df, cols):
    d = df[[ycol] + cols].dropna()
    y = d[ycol].values.astype(float)
    X = np.column_stack([np.ones(len(y))] + [d[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - (e @ e).sum() / ((y - y.mean()) ** 2).sum(), b


def loo_block_cv(df, feat_sets, block_col):
    blocks = np.unique(df[block_col].dropna().values)
    preds = {k: np.full(len(df), np.nan) for k in feat_sets}
    for b in blocks:
        tr = df[df[block_col] != b]; te = df[df[block_col] == b]
        if len(tr) < 8 or len(te) == 0:
            continue
        for name, feats in feat_sets.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in feats])
            bb, *_ = np.linalg.lstsq(Xtr, tr[ycol].values.astype(float), rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in feats])
            preds[name][te.index] = Xte @ bb
    y = df[ycol].values.astype(float)
    out = {}
    for name, p in preds.items():
        mm = ~np.isnan(p)
        ssr = ((y[mm] - p[mm]) ** 2).sum(); sst = ((y[mm] - y[mm].mean()) ** 2).sum()
        out[name] = 1 - ssr / sst
    return out


# ---- Q1: 逐水力指标单独条件增量 ----
print("=" * 90)
print("Q1: up_size 相对【单个】水力指标的条件增量（全样本 ΔR²，cl_max 分块 CV ΔR²）")
print("=" * 90)
HYD_SINGLE = [("log_tt_sum", "曼宁Στ(汇流时间)"), ("log_up_area", "Voronoi面积"),
              ("log_up_eff", "有效产流面积")]

df = m.dropna(subset=[ycol, "log_up", "cl_max"])
print(f"\n{'水力指标':16s} {'R2_hyd':>8s} {'R2+count':>8s} {'ΔR2(全样本)':>11s} "
      f"{'ΔR2(CV cl_max)':>14s}")
for hc, lab in HYD_SINGLE:
    d = df.dropna(subset=[hc])
    r_hyd, _ = ols_r2(d, BASE + [hc])
    r_full, _ = ols_r2(d, BASE + [hc, "log_up"])
    fs_d = r_full - r_hyd
    cv = loo_block_cv(d, {"hyd": BASE + [hc], "full": BASE + [hc, "log_up"]}, "cl_max")
    cv_d = cv["full"] - cv["hyd"]
    print(f"{lab:16s} {r_hyd:8.4f} {r_full:8.4f} {fs_d:+11.4f} {cv_d:+14.4f}")

# ---- Q2: 总量 vs 均值 ----
print("\n" + "=" * 90)
print("Q2: up_size 预测的是「总量 Στ」还是「均值 Στ/N」？")
print("=" * 90)
d = m.dropna(subset=["log_up", "log_tt_sum", "log_tt_mean"])
for a, b in [("log_up", "log_tt_sum"), ("log_up", "log_tt_mean")]:
    r = np.corrcoef(d[a], d[b])[0, 1]
    print(f"  corr({a}, {b}) = {r:+.4f}")

# 三模型对比：count vs Στ vs Στ/N
print(f"\n  响应={ycol}，各单变量模型（+BASE）的 ΔR²：")
for v, lab in [("log_up", "up_size(计数)"), ("log_tt_sum", "Στ(总量)"), ("log_tt_mean", "Στ/N(均值)")]:
    r0, _ = ols_r2(m, BASE)
    r1, _ = ols_r2(m, BASE + [v])
    print(f"    {lab:16s} ΔR² = {r1 - r0:+.4f}  (R²={r1:.4f})")

# 增量：count over Στ（已经知道），count over Στ/N
print(f"\n  条件增量（全样本）：")
r_tt, _ = ols_r2(m, BASE + ["log_tt_sum"])
r_tt_cnt, _ = ols_r2(m, BASE + ["log_tt_sum", "log_up"])
print(f"    count over Στ   : ΔR² = {r_tt_cnt - r_tt:+.4f}")
r_tm, _ = ols_r2(m, BASE + ["log_tt_mean"])
r_tm_cnt, _ = ols_r2(m, BASE + ["log_tt_mean", "log_up"])
print(f"    count over Στ/N : ΔR² = {r_tm_cnt - r_tm:+.4f}")
r_cnt, _ = ols_r2(m, BASE + ["log_up"])
print(f"    Στ over count   : ΔR² = {r_tt_cnt - r_cnt:+.4f}")
print(f"    Στ/N over count : ΔR² = {r_tm_cnt - r_cnt:+.4f}")

# ---- 峰值方向一致性（补充：说明这是"峰值时刻"而非"起涨时刻"）----
print("\n" + "=" * 90)
print("方向一致性：log_up 的 β 符号（全样本，cluster-robust 略）")
print("=" * 90)
for col in ["se_tpk_max", "se_tpk_half", "se_tpk_90", "se_tpk_cent", "se_tpk_firstpk"]:
    r0, b = ols_r2(m, BASE + ["log_up"])
    # β 是最后一个系数
    beta = b[-1]
    r = np.corrcoef(m.dropna(subset=[col])["log_up"], m.dropna(subset=[col])[col])[0, 1]
    print(f"  {col:16s} β={beta:+7.3f}  corr={r:+.3f}  → {'峰时刻(早)' if beta < 0 else '起涨(晚)'}")
