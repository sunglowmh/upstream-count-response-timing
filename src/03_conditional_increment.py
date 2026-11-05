# -*- coding: utf-8 -*-
"""
P1-H3：条件增量检验（新定位的判定核心）
=========================================
核心问题（承用户新定位）：up_size 与已知水力尺度指标高度相关(r≈0.87-0.89)，
  因此关键不再是"发现了什么新机制"，而是——
  在控制显式计算的汇流时间(Στ)、汇水面积(Voronoi)、有效产流面积后，
  up_size 是否仍提供「可复现」的增量解释/预测信息？

判定逻辑（H3 vs H4）：
  - 若空间分块 CV 中，up_size 在控制水力基线后仍有稳定增量 → H3 成立（独立增量）
  - 若增量在 CV 中消失 → H4 成立（up_size 只是已知水力尺度的低维代理）

方法与旧版的关键差异（严格性升级）：
  1) 响应变量用 P0 独立重建的 5 种峰值定义（主：global-max，LOO 事件残差化）
  2) 空间分块 CV 三套口径：cl_max(27 极大上游集合) / cl_seg(154 主干段) / 空间 kmeans(27)
  3) 增量显著性用「置换检验」：打乱 up_size 标签，重算 CV ΔR²，得零分布 p 值

水力基线（用户指定三项）：
  - log_tt_sum  = log1p(up_travel_sum)   曼宁汇流时间 Στ (小时)
  - log_up_area = log1p(up_area)         Voronoi 汇水面积 (m²)
  - log_up_eff  = log1p(up_eff_area)     有效产流面积 (m²)

输出：
  - p1_h3_master.csv              主合并表（站级）
  - p1_h3_fullsample.csv          全样本条件 ΔR² + cluster-robust SE
  - p1_h3_cv.csv                  空间分块 CV 的 out-of-fold R² 与 ΔR²
  - p1_h3_permutation.csv         置换检验零分布
"""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.cluster.vq import kmeans2
import io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()

PY = "E:/work/workBuddy/BlueHeart"
T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))


def load(name):
    df = pd.read_csv(f"{T}/{name}", encoding="utf-8-sig")
    for c in df.columns:
        if c.strip() in ("st", "node_id"):
            df[c] = df[c].astype(str).str.strip()
    return df


# =====================================================================
# 1. 构造主表：水力指标 + 聚类 + 坐标 + system
# =====================================================================
print("=" * 80)
print("1. 构造主表")
print("=" * 80)

k = load("m13k_runoff_regression.csv")          # up_size, up_area, up_eff_area, (old st_effect)
o = load("m13o_travel_time.csv")                # up_travel_sum, up_travel_mean
cl = load("m13m_clustered_se.csv")              # cl_wcc, cl_max, cl_seg
typ = load("m3c_station_typology.csv")          # system, node_id, x, y, phys_cls, has_pump
phys = load("m3e_station_physics.csv")          # phys_cls, has_pump, avg_diameter(虚设)

master = k[["st", "node_id", "up_size", "up_area", "up_eff_area"]].copy()
master = master.merge(o[["st", "up_travel_sum", "up_travel_mean"]], on="st", how="left")
master = master.merge(cl[["st", "cl_wcc", "cl_max", "cl_seg"]], on="st", how="left")
master = master.merge(typ[["st", "system", "x", "y", "phys_cls", "has_pump"]], on="st", how="left")

# log 尺度
for col, new in [("up_size", "log_up"), ("up_area", "log_up_area"),
                 ("up_eff_area", "log_up_eff"), ("up_travel_sum", "log_tt_sum"),
                 ("up_travel_mean", "log_tt_mean")]:
    master[new] = np.log1p(pd.to_numeric(master[col], errors="coerce").fillna(0).values)

# system 哑变量
master["sys_foul"] = (master["system"] == "foul").astype(float).values
master["sys_combined"] = (master["system"] == "combined").astype(float).values

# 坐标标准化
master["x_std"] = (master["x"] - master["x"].mean()) / master["x"].std()
master["y_std"] = (master["y"] - master["y"].mean()) / master["y"].std()

# =====================================================================
# 2. 新响应变量：从 p0 独立重建的 5 种峰值定义算 st_effect（LOO 事件残差化）
# =====================================================================
print("2. 重建新响应变量 st_effect（LOO 事件残差化）")
D = pd.read_csv(f"{T}/p0_station_event_tpk_multidef.csv", encoding="utf-8-sig")
D["st"] = D["st"].astype(int)
DEFS = [("tpk_max", "global max"), ("tpk_half", "half-peak"), ("tpk_90", "90%-rise"),
        ("tpk_cent", "centroid"), ("tpk_firstpk", "first-sig-peak")]


def st_effect_of(col):
    g = D[["st", "ev_id", col]].dropna(subset=[col]).copy()
    g["t_h"] = g[col] * 0.25
    ev_sum = g.groupby("ev_id")["t_h"].transform("sum")
    ev_cnt = g.groupby("ev_id")["t_h"].transform("count")
    ev_mean_loo = (ev_sum - g["t_h"]) / (ev_cnt - 1)
    g["resid"] = g["t_h"] - ev_mean_loo
    return g.groupby("st")["resid"].mean()


master["st_int"] = master["st"].astype(int)
for col, lab in DEFS:
    master[f"se_{col}"] = master["st_int"].map(st_effect_of(col))

master = master.dropna(subset=["se_tpk_max"]).reset_index(drop=True)
print(f"  n = {len(master)} 站（含新响应变量）")
print(f"  up_size 唯一值 = {master.up_size.nunique()}")

# =====================================================================
# 3. H2：代理关系（up_size 与各水力指标的 log-log 相关）
# =====================================================================
print("\n" + "=" * 80)
print("3. H2 代理关系：up_size 与水力指标的 log-log 相关")
print("=" * 80)
h2_cols = ["log_tt_sum", "log_tt_mean", "log_up_area", "log_up_eff"]
for c in h2_cols:
    m = master[["log_up", c]].dropna()
    r = np.corrcoef(m["log_up"], m[c])[0, 1]
    print(f"  corr(log_up, {c:12s}) = {r:+.4f}  (n={len(m)})")

# =====================================================================
# 4. H3：全样本条件 ΔR² + cluster-robust SE
# =====================================================================
print("\n" + "=" * 80)
print("4. H3 全样本条件增量（OLS ΔR² + cluster-robust SE, 聚类=cl_max）")
print("=" * 80)

BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
HYD = ["log_tt_sum", "log_up_area", "log_up_eff"]


def ols(df, cols, ycol):
    d = df[[ycol] + cols].dropna()
    y = d[ycol].values.astype(float)
    X = np.column_stack([np.ones(len(y))] + [d[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1 - (e @ e).sum() / ((y - y.mean()) ** 2).sum()
    return b, e, X, y, d.index.values


def cluster_se_from(components):
    """components: (b, e, X, y, groups). 返回每个系数的 se/t/p"""
    b, e, X, y, groups = components
    n, k = X.shape
    G = len(np.unique(groups))
    B = np.zeros((k, k))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        Xg = X[idx]; eg = e[idx]
        B += Xg.T @ (np.outer(eg, eg) @ Xg)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ B @ XtXi
    V *= (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), G - 1))
    return se, t, p, G


def full_conditional(ycol):
    """返回 dict: 各模型的 R² 与关键增量"""
    res = {}
    m_base = master[["st", ycol] + BASE + HYD + ["log_up", "cl_max"]].dropna()
    r0 = ols(m_base, BASE, ycol)
    r_hyd = ols(m_base, BASE + HYD, ycol)
    r_cnt = ols(m_base, BASE + ["log_up"], ycol)
    r_full = ols(m_base, BASE + HYD + ["log_up"], ycol)
    res["R2_base"] = r0[0] ** 0
    # R²
    def r2_of(r):
        b, e, X, y, _ = r
        return 1 - (e @ e).sum() / ((y - y.mean()) ** 2).sum()
    res["R2_base"] = r2_of(r0)
    res["R2_hyd"] = r2_of(r_hyd)
    res["R2_cnt"] = r2_of(r_cnt)
    res["R2_full"] = r2_of(r_full)
    res["dR2_count_over_hyd"] = res["R2_full"] - res["R2_hyd"]
    res["dR2_hyd_over_count"] = res["R2_full"] - res["R2_cnt"]
    # cluster-robust SE 对 log_up（在 full 模型里）与 log_tt_sum 等
    groups = m_base["cl_max"].values
    se, t, p, G = cluster_se_from((r_full[0], r_full[1], r_full[2], r_full[3], groups))
    cols_full = BASE + HYD + ["log_up"]
    i_up = cols_full.index("log_up")
    res["log_up_beta"] = r_full[0][i_up + 1]
    res["log_up_t_cluster"] = t[i_up + 1]
    res["log_up_p_cluster"] = p[i_up + 1]
    res["n"] = len(m_base)
    res["G"] = G
    return res


print(f"{'响应变量':16s} {'R2_base':>8s} {'R2_hyd':>8s} {'R2_full':>8s} "
      f"{'ΔR2_count|hyd':>13s} {'ΔR2_hyd|count':>13s} {'β_up':>7s} {'t_cl':>6s} {'p_cl':>6s} {'n':>4s}")
for col, lab in DEFS:
    ycol = f"se_{col}"
    r = full_conditional(ycol)
    print(f"{lab:16s} {r['R2_base']:8.4f} {r['R2_hyd']:8.4f} {r['R2_full']:8.4f} "
          f"{r['dR2_count_over_hyd']:+13.4f} {r['dR2_hyd_over_count']:+13.4f} "
          f"{r['log_up_beta']:+7.3f} {r['log_up_t_cluster']:+6.2f} {r['log_up_p_cluster']:6.4f} {r['n']:4d}")

# =====================================================================
# 5. H3：空间分块交叉验证
# =====================================================================
print("\n" + "=" * 80)
print("5. H3 空间分块 CV（out-of-fold R² 与 ΔR²）")
print("=" * 80)


def loo_block_cv(df, feat_sets, ycol, block_col):
    blocks = np.unique(df[block_col].dropna().values)
    preds = {name: np.full(len(df), np.nan) for name in feat_sets}
    for b in blocks:
        tr = df[df[block_col] != b]
        te = df[df[block_col] == b]
        if len(tr) < 8 or len(te) == 0:
            continue
        for name, feats in feat_sets.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in feats])
            ytr = tr[ycol].values.astype(float)
            bb, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in feats])
            preds[name][te.index] = Xte @ bb
    y = df[ycol].values.astype(float)
    out = {}
    for name, p in preds.items():
        m = ~np.isnan(p)
        if m.sum() < 10:
            out[name] = np.nan
            continue
        ssr = ((y[m] - p[m]) ** 2).sum()
        sst = ((y[m] - y[m].mean()) ** 2).sum()
        out[name] = 1 - ssr / sst
    return out


# 空间 kmeans 分块
xy = master[["x", "y"]].values.astype(float)
for kk, tag in [(27, "sp_k27"), (10, "sp_k10")]:
    cen, lab = kmeans2(xy, kk, minit="points", seed=0)
    master[tag] = lab

FEATS = {
    "base": BASE,
    "hyd": BASE + HYD,
    "count": BASE + ["log_up"],
    "full": BASE + HYD + ["log_up"],
}

ycol = "se_tpk_max"   # 主响应：global max
print(f"\n主响应 global-max（{ycol}），各分块口径 out-of-fold R²：")
print(f"{'分块口径':14s} {'R2_base':>9s} {'R2_hyd':>9s} {'R2_count':>9s} {'R2_full':>9s} "
      f"{'Δcount|hyd':>10s} {'Δhyd|count':>10s}")
cv_rows = []
for block, lab in [("cl_max", "cl_max(27)"), ("cl_seg", "cl_seg(154)"),
                   ("sp_k27", "spatial_k27"), ("sp_k10", "spatial_k10")]:
    df = master.dropna(subset=[ycol, "log_up", "log_tt_sum", "log_up_area", "log_up_eff", block])
    r = loo_block_cv(df, FEATS, ycol, block)
    dc = r["full"] - r["hyd"]
    dh = r["full"] - r["count"]
    print(f"{lab:14s} {r['base']:9.4f} {r['hyd']:9.4f} {r['count']:9.4f} {r['full']:9.4f} "
          f"{dc:+10.4f} {dh:+10.4f}")
    cv_rows.append({"block": block, "n_block": df[block].nunique(),
                    **{f"R2_{k}": v for k, v in r.items()},
                    "dR2_count_over_hyd": dc, "dR2_hyd_over_count": dh})

# =====================================================================
# 6. 置换检验（up_size 标签打乱，重算 CV ΔR²）
# =====================================================================
print("\n" + "=" * 80)
print("6. 置换检验：ΔR²(count over hyd) 在 cl_max 分块 CV 下的可复现性")
print("=" * 80)
df0 = master.dropna(subset=[ycol, "log_up", "log_tt_sum", "log_up_area", "log_up_eff", "cl_max"]).copy()
obs = loo_block_cv(df0, FEATS, ycol, "cl_max")
obs_d = obs["full"] - obs["hyd"]
print(f"  观测 ΔR²(count over hyd) = {obs_d:+.4f}")

rng = np.random.default_rng(0)
n_perm = 300
perm_d = []
for it in range(n_perm):
    dfp = df0.copy()
    dfp["log_up"] = rng.permutation(df0["log_up"].values)
    rp = loo_block_cv(dfp, FEATS, ycol, "cl_max")
    perm_d.append(rp["full"] - rp["hyd"])
perm_d = np.array(perm_d)
pval = (np.sum(perm_d >= obs_d) + 1) / (n_perm + 1)
print(f"  置换零分布: mean={perm_d.mean():+.4f}, P2.5={np.percentile(perm_d,2.5):+.4f}, "
      f"P97.5={np.percentile(perm_d,97.5):+.4f}")
print(f"  单侧 p(置换 ΔR² ≥ 观测) = {pval:.4f}  → {'可复现增量(H3)' if pval < 0.05 else '不可复现(H4)'}")

# =====================================================================
# 7. 保存
# =====================================================================
master.to_csv(f"{T}/p1_h3_master.csv", index=False, encoding="utf-8-sig")
cvdf = pd.DataFrame(cv_rows)
cvdf.to_csv(f"{T}/p1_h3_cv.csv", index=False, encoding="utf-8-sig")
perm_s = pd.DataFrame({"perm_delta": perm_d})
perm_s.to_csv(f"{T}/p1_h3_permutation.csv", index=False, encoding="utf-8-sig")

# 全样本表
fs_rows = []
for col, lab in DEFS:
    r = full_conditional(f"se_{col}")
    r["response"] = lab
    fs_rows.append(r)
pd.DataFrame(fs_rows).to_csv(f"{T}/p1_h3_fullsample.csv", index=False, encoding="utf-8-sig")

print(f"\n已保存 p1_h3_master.csv / p1_h3_cv.csv / p1_h3_fullsample.csv / p1_h3_permutation.csv")
print(f"总用时 {time.time()-t0:.1f}s")
