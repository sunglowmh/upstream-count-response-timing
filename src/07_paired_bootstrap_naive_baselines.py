# -*- coding: utf-8 -*-
"""
R1 + R3：成对 bootstrap 与朴素基线
==================================
R1  计数 vs 三指标块 / 计数 vs Sigma-tau 的**成对**差值区间
    （重抽站点、保留折结构；pooled out-of-fold R^2 的差，而非两个独立点估计）
R3  朴素基线（最近站 / 5 近邻 IDW / 训练折均值 / 子区均值）在同一折、同一
    fold-consistent 目标上的表现；以及时间外推（前段事件拟合、后段事件检验）

所有折构造与目标重建与 p0b_foldref_cv.py 完全一致（训练折事件参照，
训练站 LOO、测试站用完整训练折参照），以便与 Table 2 对得上。

输出：
  p1_paired_boot.csv     各 blocking 的 R^2、成对差值及其 bootstrap 区间
  p1_naive_baseline.csv  朴素基线的 OOF R^2 与 pooled Spearman
  p1_oof_pred.csv        逐站 OOF 预测（复核用）
  p1_time_split.csv      时间外推结果
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from scipy import stats
import io, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()

T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
m["st"] = m["st"].astype(str).str.strip()
m["st_int"] = m["st"].astype(int)
m = m.reset_index(drop=True)

D = pd.read_csv(f"{T}/p0_station_event_tpk_multidef.csv", encoding="utf-8-sig")
D["st"] = D["st"].astype(int)
D = D[D["st"].isin(set(m["st_int"]))].reset_index(drop=True)

COL = "tpk_max"
BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
HYD = ["log_tt_sum", "log_up_area", "log_up_eff"]
FEATS = {"base": BASE, "hyd": BASE + HYD, "count": BASE + ["log_up"],
         "full": BASE + HYD + ["log_up"], "tau": BASE + ["log_tt_sum"],
         "ctau": BASE + ["log_tt_sum", "log_up"]}

OBS = D[["st", "ev_id", COL]].dropna(subset=[COL]).copy()
OBS["t_h"] = OBS[COL].values * 0.25


def fold_outcome(train_sts, obs=OBS):
    """训练折事件参照；训练站 LOO，非训练站用完整训练折参照。"""
    is_tr = obs["st"].isin(train_sts).values
    sub = obs[is_tr]
    s_sum = sub.groupby("ev_id")["t_h"].sum()
    n_cnt = sub.groupby("ev_id")["t_h"].size()
    ref = (s_sum / n_cnt).rename("ref")
    g = obs.merge(ref, left_on="ev_id", right_index=True, how="left")
    g = g.merge(n_cnt.rename("n_tr"), left_on="ev_id", right_index=True, how="left")
    g = g[g["ref"].notna() & (g["n_tr"] >= 1)].copy()
    own = g["st"].isin(train_sts).values
    ok_tr = own & (g["n_tr"].values >= 2)
    resid = np.where(ok_tr,
                     g["t_h"].values - (g["n_tr"].values * g["ref"].values - g["t_h"].values)
                     / (g["n_tr"].values - 1),
                     g["t_h"].values - g["ref"].values)
    g["resid"] = resid
    g = g[ok_tr | (~own)]
    return g.groupby("st")["resid"].mean()


Y_CHECK = fold_outcome(set(m["st_int"]))
print(f"[自检] 全样本参照 vs master.se_tpk_max 最大绝对差 = "
      f"{(m['st_int'].map(Y_CHECK) - m['se_tpk_max']).abs().max():.3e}")

# 节点级 kmeans（与 Table 2 一致）
node_coords = m.groupby("node_id")[["x", "y"]].mean()
for kk in (27, 10):
    _, lab = kmeans2(node_coords.values.astype(float), kk, minit="points", seed=0)
    m[f"sp_k{kk}_node"] = m["node_id"].map(pd.Series(lab, index=node_coords.index)).values
# k=5 子区（朴素基线用）
_, lab5 = kmeans2(node_coords.values.astype(float), 5, minit="points", seed=0)
m["sub5"] = m["node_id"].map(pd.Series(lab5, index=node_coords.index)).values

BLOCKS = [("cl_max", "maximal upstream sets (26)"),
          ("cl_seg", "trunk segments (154)"),
          ("sp_k27_node", "spatial k-means k=27"),
          ("sp_k10_node", "spatial k-means k=10")]


# ---------------------------------------------------------------- 逐折预测
def run_block(block_col):
    """返回逐站记录：st, y, pred_<spec>, pred_<naive>"""
    recs = []
    for b in np.unique(m[block_col].dropna().values):
        te_idx = m.index[m[block_col] == b].to_numpy()
        tr_idx = m.index[m[block_col] != b].to_numpy()
        if len(tr_idx) < 8 or len(te_idx) == 0:
            continue
        y = fold_outcome(set(m.loc[tr_idx, "st_int"]))
        ytr = m.loc[tr_idx, "st_int"].map(y).to_numpy(float)
        yte = m.loc[te_idx, "st_int"].map(y).to_numpy(float)
        ok = ~np.isnan(ytr)
        if ok.sum() < 8:
            continue
        tr_idx = tr_idx[ok]; ytr = ytr[ok]
        good = ~np.isnan(yte)
        te_idx = te_idx[good]; yte = yte[good]
        if len(te_idx) == 0:
            continue

        tr, te = m.loc[tr_idx], m.loc[te_idx]
        rec = {"block": b, "te": te_idx, "y": yte, "n_te": len(te_idx)}

        # --- 参数模型（4 个规格）
        for name, feats in FEATS.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in feats])
            bb, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in feats])
            rec[f"pred_{name}"] = Xte @ bb

        # --- ridge（惩罚系数由训练站内部 5 折 CV 选取）
        for name in ("count", "hyd"):
            rec[f"pred_r{name}"] = _ridge_fit_predict(tr, ytr, te, FEATS[name])

        # --- 朴素基线
        tr_xy = tr[["x", "y"]].to_numpy(float)
        te_xy = te[["x", "y"]].to_numpy(float)
        tree = cKDTree(tr_xy)
        dist, idx = tree.query(te_xy, k=min(5, len(tr)))
        dist = np.atleast_2d(dist.T).T if dist.ndim == 1 else dist
        idx = np.atleast_2d(idx.T).T if idx.ndim == 1 else idx
        ytr_v = ytr
        rec["pred_near"] = ytr_v[idx[:, 0]]
        w = 1.0 / np.maximum(dist, 1e-6) ** 2
        rec["pred_idw5"] = (ytr_v[idx] * w).sum(1) / w.sum(1)
        rec["pred_mean"] = np.full(len(te), ytr_v.mean())
        sub_mean = pd.Series(ytr_v, index=tr["sub5"].values).groupby(level=0).mean()
        glob = ytr_v.mean()
        rec["pred_sub5"] = te["sub5"].map(sub_mean).fillna(glob).to_numpy(float)
        # 协变量调整版：把朴素预测当作一个 predictor 加入 base 模型
        for nm in ("near", "idw5"):
            # 训练侧需要训练站自己的朴素预测（折内 LOO，避免用自身）
            p_tr = _train_self(nm, tr, ytr_v)
            Xtr2 = np.column_stack([np.ones(len(tr))] +
                                   [tr[c].to_numpy(float) for c in BASE] + [p_tr])
            bb2, *_ = np.linalg.lstsq(Xtr2, ytr, rcond=None)
            Xte2 = np.column_stack([np.ones(len(te))] +
                                   [te[c].to_numpy(float) for c in BASE] + [rec["pred_" + nm]])
            rec[f"pred_{nm}_adj"] = Xte2 @ bb2
        recs.append(rec)
    return recs


ALPHAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def _ridge_solve(Z, y, a):
    Zc = Z - Z.mean(0)
    yc = y - y.mean()
    return np.linalg.solve(Zc.T @ Zc + a * np.eye(Z.shape[1]), Zc.T @ yc)


def _ridge_fit_predict(tr, ytr, te, feats):
    """标准化后用训练站内部 5 折 CV 选 alpha（不接触测试折）。"""
    Xtr = tr[feats].to_numpy(float)
    Xte = te[feats].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    n = len(ytr)
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(n), 5)
    best = (np.inf, ALPHAS[0])
    for a in ALPHAS:
        sse = 0.0
        for f in parts:
            msk = np.ones(n, bool); msk[f] = False
            b = _ridge_solve(Ztr[msk], ytr[msk], a)
            p = Ztr[f] @ b + (ytr[msk].mean() - Ztr[msk].mean(0) @ b)
            sse += ((ytr[f] - p) ** 2).sum()
        if sse < best[0]:
            best = (sse, a)
    b = _ridge_solve(Ztr, ytr, best[1])
    return Zte @ b + (ytr.mean() - Ztr.mean(0) @ b)


def _train_self(nm, tr, ytr_v):
    """训练站的朴素预测：用训练站内部的留一（最近邻/IDW），不用自身。"""
    xy = tr[["x", "y"]].to_numpy(float)
    tree = cKDTree(xy)
    d, ix = tree.query(xy, k=min(6, len(tr)))
    d = np.atleast_2d(d if d.ndim > 1 else d[None, :]).reshape(len(xy), -1)
    ix = np.atleast_2d(ix if ix.ndim > 1 else ix[None, :]).reshape(len(xy), -1)
    if nm == "near":
        return ytr_v[ix[:, 1]]
    w = 1.0 / np.maximum(d[:, 1:], 1e-6) ** 2
    return (ytr_v[ix[:, 1:]] * w).sum(1) / w.sum(1)


def pooled(recs, key):
    y = np.concatenate([r["y"] for r in recs])
    p = np.concatenate([r[key] for r in recs])
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], p[ok]
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - p) ** 2).sum() / sst
    rho = stats.spearmanr(y, p).statistic
    return r2, rho, y, p


def pair_boot(recs, keyA, keyB, B=4000, seed=7):
    """成对 bootstrap：重抽站点（保持每站在其折内的 y 与两个预测），
    重算 pooled R^2，返回 Δ=R2_A-R2_B 的区间与 P(Δ>0)。"""
    y = np.concatenate([r["y"] for r in recs])
    a = np.concatenate([r[keyA] for r in recs])
    b = np.concatenate([r[keyB] for r in recs])
    ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
    y, a, b = y[ok], a[ok], b[ok]
    n = len(y)
    d0 = _dr2(y, a, b)
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for i in range(B):
        ii = rng.integers(0, n, n)
        draws[i] = _dr2(y[ii], a[ii], b[ii])
    return d0, np.percentile(draws, 2.5), np.percentile(draws, 97.5), (draws > 0).mean(), n


def _dr2(y, a, b):
    sst = ((y - y.mean()) ** 2).sum()
    ra = 1 - ((y - a) ** 2).sum() / sst
    rb = 1 - ((y - b) ** 2).sum() / sst
    return ra - rb


# ---------------------------------------------------------------- 主循环
print("\n" + "=" * 96)
print("A. 逐 blocking：pooled OOF R^2（复核 Table 2）")
print("=" * 96)
allrecs = {}
rows = []
for col, lab in BLOCKS:
    recs = run_block(col)
    allrecs[lab] = recs
    r2 = {k: pooled(recs, "pred_" + k)[0] for k in FEATS}
    print(f"\n{lab}   (folds={len(recs)}, n={sum(r['n_te'] for r in recs)})")
    print("   " + "  ".join(f"{k}={v:+.4f}" for k, v in r2.items()))
    rows.append({"blocking": lab, **{f"R2_{k}": v for k, v in r2.items()}})

print("\n" + "=" * 96)
print("B. 成对 bootstrap（重抽站点，B=4000）：ΔR^2 = R^2_A - R^2_B")
print("=" * 96)
boot_rows = []
pairs = [("count", "hyd", "count − three-indicator block"),
         ("count", "tau", "count − Sigma-tau alone"),
         ("count", "base", "count − baseline covariates"),
         ("full", "hyd", "conditional increment (count | hyd, 3 indicators)"),
         ("ctau", "tau", "conditional increment (count | Sigma-tau)")]
for col, lab in BLOCKS:
    recs = allrecs[lab]
    print(f"\n{lab}")
    for a, b, desc in pairs:
        d, lo, hi, fr, n = pair_boot(recs, "pred_" + a, "pred_" + b)
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"   {desc:40s} Δ={d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={fr:.3f}{star}")
        boot_rows.append({"blocking": lab, "contrast": desc, "delta": d,
                          "lo": lo, "hi": hi, "p_gt0": fr, "n": n})

# ---------------------------------------------------------------- 朴素基线
print("\n" + "=" * 96)
print("C. 朴素基线（同折、同 fold-consistent 目标）")
print("=" * 96)
nb_rows = []
for col, lab in BLOCKS:
    recs = allrecs[lab]
    print(f"\n{lab}")
    for key, nm in [("pred_mean", "train-fold mean (intercept only)"),
                    ("pred_sub5", "subregion mean (k=5)"),
                    ("pred_near", "nearest monitored station"),
                    ("pred_idw5", "IDW-5 monitored stations"),
                    ("pred_near_adj", "nearest + spatial/system covariates"),
                    ("pred_idw5_adj", "IDW-5 + spatial/system covariates"),
                    ("pred_count", "upstream count + covariates"),
                    ("pred_tau", "Sigma-tau + covariates")]:
        r2, rho, _, _ = pooled(recs, key)
        print(f"   {nm:38s} R²={r2:+.4f}  pooled Spearman={rho:+.3f}")
        nb_rows.append({"blocking": lab, "baseline": nm, "R2": r2, "spearman": rho})

print("\n" + "=" * 96)
print("D. 计数 vs 朴素基线的成对 bootstrap")
print("=" * 96)
for col, lab in BLOCKS:
    recs = allrecs[lab]
    print(f"\n{lab}")
    for key, nm in [("pred_near", "nearest station"), ("pred_idw5", "IDW-5"),
                    ("pred_sub5", "subregion mean (k=5)")]:
        d, lo, hi, fr, n = pair_boot(recs, "pred_count", key)
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"   count − {nm:22s} Δ={d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={fr:.3f}{star}")
        boot_rows.append({"blocking": lab, "contrast": f"count − {nm}", "delta": d,
                          "lo": lo, "hi": hi, "p_gt0": fr, "n": n})

# 保存
pd.DataFrame(rows).to_csv(f"{T}/p1_paired_boot_r2.csv", index=False)

print("\n" + "=" * 96)
print("D2. ridge（训练折内部 CV 选 alpha）：R^2 与成对区间")
print("=" * 96)
for col, lab in BLOCKS:
    recs = allrecs[lab]
    r_c = pooled(recs, "pred_rcount")[0]
    r_b = pooled(recs, "pred_rhyd")[0]
    d, lo, hi, fr, n = pair_boot(recs, "pred_rcount", "pred_rhyd")
    print(f"   {lab:32s} count={r_c:+.4f} block={r_b:+.4f}  "
          f"Δ={d:+.4f} [{lo:+.4f}, {hi:+.4f}] P(Δ>0)={fr:.3f}")
    boot_rows.append({"blocking": lab, "contrast": "ridge: count − block",
                      "delta": d, "lo": lo, "hi": hi, "p_gt0": fr, "n": n})
    for rr in rows:
        if rr["blocking"] == lab:
            rr["R2_ridge_count"] = r_c
            rr["R2_ridge_block"] = r_b
pd.DataFrame(rows).to_csv(f"{T}/p1_paired_boot_r2.csv", index=False)
pd.DataFrame(boot_rows).to_csv(f"{T}/p1_paired_boot.csv", index=False)
pd.DataFrame(nb_rows).to_csv(f"{T}/p1_naive_baseline.csv", index=False)

# 逐站 OOF 预测（取 k=27，规模适中）
recs = allrecs["spatial k-means k=27"]
out = []
for r in recs:
    for j, st in enumerate(m.loc[r["te"], "st_int"].values):
        row = {"st": st, "y": r["y"][j]}
        for k in list(FEATS) + ["near", "idw5", "mean", "sub5"]:
            row[k] = float(r["pred_" + k][j])
        out.append(row)
pd.DataFrame(out).to_csv(f"{T}/p1_oof_pred.csv", index=False)

# ---------------------------------------------------------------- 时间外推
print("\n" + "=" * 96)
print("E. 时间外推：前段事件拟合，后段事件检验")
print("=" * 96)
ev = pd.read_csv(f"{T}/m1a_events.csv", encoding="utf-8-sig")
grid0 = pd.Timestamp("2024-02-19", tz="UTC")
ev["date"] = grid0 + pd.to_timedelta(ev["s"].astype(int) * 15, unit="min")
evd = ev[["ev_id", "date"]].copy() if "ev_id" in ev.columns else None
if evd is None:
    ev = ev.reset_index().rename(columns={"index": "ev_id"})
    evd = ev[["ev_id", "date"]].copy()
evd = evd[evd.ev_id.isin(OBS.ev_id.unique())]
med = evd["date"].median()
early = set(evd.loc[evd.date <= med, "ev_id"])
late = set(evd.loc[evd.date > med, "ev_id"])
print(f"   事件中位日期 {med.date()}；前段 {len(early)} 个事件，后段 {len(late)} 个")

OBS_e = OBS[OBS.ev_id.isin(early)]
OBS_l = OBS[OBS.ev_id.isin(late)]
alls = set(OBS["st"])
se = fold_outcome(alls, OBS_e)
sl = fold_outcome(alls, OBS_l)
tmp = pd.DataFrame({"st": sorted(alls)})
tmp["se"] = tmp["st"].map(se); tmp["sl"] = tmp["st"].map(sl)
tmp = tmp.merge(m[["st_int", "log_up", "x_std", "y_std", "sys_foul", "sys_combined"]],
                left_on="st", right_on="st_int", how="left")
tmp = tmp.dropna(subset=["se", "sl"])
print(f"   两段都有有效目标的站 = {len(tmp)}/{len(alls)}")
r_stab = stats.pearsonr(tmp.se, tmp.sl)
print(f"   corr(s_i 前段, s_i 后段) = {r_stab.statistic:+.3f}  (p={r_stab.pvalue:.2g})"
      f"  ← 时间稳定性天花板 R²≈{r_stab.statistic**2:.3f}")

X = np.column_stack([np.ones(len(tmp))] + [tmp[c].to_numpy(float) for c in BASE + ["log_up"]])
b, *_ = np.linalg.lstsq(X, tmp["se"].to_numpy(float), rcond=None)
pred = X @ b
y = tmp["sl"].to_numpy(float)
r2_time = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rho = stats.spearmanr(y, pred).statistic
print(f"   前段拟合 → 后段预测：R²={r2_time:+.4f}  pooled Spearman={rho:+.3f}")
pred_tr = X @ b
r2_in = 1 - ((tmp['se'] - pred_tr) ** 2).sum() / ((tmp['se'] - tmp['se'].mean()) ** 2).sum()
print(f"   （同一模型在前段的拟合 R²={r2_in:+.4f}，用于对照退化幅度）")

pd.DataFrame([{"n": len(tmp), "corr_early_late": r_stab.statistic,
               "R2_early_to_late": r2_time, "spearman_early_to_late": rho,
               "R2_in_sample_early": r2_in, "median_event_date": str(med.date()),
               "n_events_early": len(early), "n_events_late": len(late)}]
             ).to_csv(f"{T}/p1_time_split.csv", index=False)

print(f"\n完成，用时 {time.time()-t0:.1f}s")
