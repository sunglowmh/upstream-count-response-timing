# -*- coding: utf-8 -*-
"""
补充：无泄漏口径（折内重建训练目标）下的 block-preserving 置换检验
====================================================================
对 cl_max 分块，观测 ΔR²(count over hyd) 用「折内重建训练目标」计算；
置换时在每个 cl_max 块内独立打乱 log_up（保持拓扑块结构），重算 ΔR²。
训练目标 ytr 只依赖 tpk 数据与训练站点集合，不依赖 log_up 标签，
故预计算一次、置换时复用，大幅加速。
"""
import os
import numpy as np
import pandas as pd
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()

PY = "E:/work/workBuddy/BlueHeart"
T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))

m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
for c in ["st", "node_id"]:
    m[c] = m[c].astype(str).str.strip()
m["st_int"] = m["st"].astype(int)

D = pd.read_csv(f"{T}/p0_station_event_tpk_multidef.csv", encoding="utf-8-sig")
D["st"] = D["st"].astype(int)

COL = "tpk_max"
BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
HYD = ["log_tt_sum", "log_up_area", "log_up_eff"]


def effect_from(df):
    g = df[["st", "ev_id", COL]].dropna(subset=[COL]).copy()
    g["t_h"] = g[COL].values * 0.25
    ev_sum = g.groupby("ev_id")["t_h"].transform("sum")
    ev_cnt = g.groupby("ev_id")["t_h"].transform("count")
    ev_mean_loo = (ev_sum - g["t_h"]) / (ev_cnt - 1)
    g["resid"] = g["t_h"] - ev_mean_loo
    return g.groupby("st")["resid"].mean()


Y_FULL = effect_from(D)

df0 = m.dropna(subset=["log_up", "log_tt_sum", "log_up_area", "log_up_eff", "cl_max"]).copy()
blocks = np.unique(df0["cl_max"].values)

# 预计算每个 block 的折内重建训练响应
ytr_map = {}
for b in blocks:
    train_sts = set(df0[df0["cl_max"] != b]["st_int"].values)
    ytr_map[b] = effect_from(D[D["st"].isin(train_sts)])
print(f"预计算 {len(blocks)} 个 block 的折内重建训练响应 完成（{time.time()-t0:.1f}s）")


def pooled_delta(log_up_vals):
    """给定 log_up 标签，返回 pooled out-of-fold ΔR² = R²(full) - R²(hyd)。"""
    pred_hyd = np.full(len(df0), np.nan)
    pred_full = np.full(len(df0), np.nan)
    y_true = np.full(len(df0), np.nan)
    for b in blocks:
        tr = df0[df0["cl_max"] != b]
        te = df0[df0["cl_max"] == b]
        ytr = tr["st_int"].map(ytr_map[b]).values.astype(float)
        ok = ~np.isnan(ytr)
        tr2, ytr2 = tr[ok], ytr[ok]
        if len(tr2) < 8:
            continue
        yte = te["st_int"].map(Y_FULL).values.astype(float)
        y_true[te.index.values] = yte
        # hyd 模型（不含 log_up，置换中不变）
        Xh_tr = np.column_stack([np.ones(len(tr2))] + [tr2[c].values for c in BASE + HYD])
        bh, *_ = np.linalg.lstsq(Xh_tr, ytr2, rcond=None)
        Xh_te = np.column_stack([np.ones(len(te))] + [te[c].values for c in BASE + HYD])
        pred_hyd[te.index.values] = Xh_te @ bh
        # full 模型（含 log_up，标签由参数给定）
        Xf_tr = np.column_stack([np.ones(len(tr2))] + [tr2[c].values for c in BASE + HYD]
                                + [log_up_vals[tr2.index.values]])
        bf, *_ = np.linalg.lstsq(Xf_tr, ytr2, rcond=None)
        Xf_te = np.column_stack([np.ones(len(te))] + [te[c].values for c in BASE + HYD]
                                + [log_up_vals[te.index.values]])
        pred_full[te.index.values] = Xf_te @ bf

    def r2(p):
        mm = ~np.isnan(p) & ~np.isnan(y_true)
        ssr = ((y_true[mm] - p[mm]) ** 2).sum()
        sst = ((y_true[mm] - y_true[mm].mean()) ** 2).sum()
        return 1 - ssr / sst

    return r2(pred_full) - r2(pred_hyd)


log_up_true = df0["log_up"].values.astype(float)
obs_d = pooled_delta(log_up_true)
print(f"观测 ΔR²(count over hyd, cl_max, 无泄漏) = {obs_d:+.4f}")

rng = np.random.default_rng(0)
n_perm = 300
perm_d = np.zeros(n_perm)
for it in range(n_perm):
    shuffled = np.empty(len(df0))
    for b in blocks:
        idx = df0.index[df0["cl_max"] == b].values
        shuffled[idx] = rng.permutation(log_up_true[idx])
    perm_d[it] = pooled_delta(shuffled)

pval = (np.sum(perm_d >= obs_d) + 1) / (n_perm + 1)
print(f"block-preserving 置换零分布: mean={perm_d.mean():+.4f}, "
      f"P2.5={np.percentile(perm_d, 2.5):+.4f}, P97.5={np.percentile(perm_d, 97.5):+.4f}")
print(f"单侧 p(置换 ΔR² ≥ 观测 {obs_d:+.4f}) = {pval:.4f}")
print(f"总用时 {time.time()-t0:.1f}s")
