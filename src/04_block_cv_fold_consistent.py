# -*- coding: utf-8 -*-
"""
P4-1 空间分块 CV 重新设计：单一折内参照目标（train-reference outcome）
=====================================================================
背景（审稿意见 R1）
  上一版 p0_leakfree_cv.py 只把「训练目标」改为折内重建，但「测试目标」仍用全样本
  leave-one-station-out 事件残差。于是
    训练目标 = 相对「训练折站点事件均值」
    测试目标 = 相对「全样本（除本站外）事件均值」——其中含其他测试站观测
  两者不是同一估计量，out-of-fold R² 不能解释为「对同一目标的空间预测」，也不足以
  支撑「无目标构造泄漏」。

本版修正（部署一致的单一参照）
  对每个折 f：只用训练站 T_f 估计事件参照  ref_j^(f) = mean_{k in T_f, 站k观测到事件j} t_kj
    * 训练站 i 的目标：s_i^(f) = mean_{j in J_i} [ t_ij - ref_j^(f, -i) ]，ref 去掉 i 自身（站自身永不进自身参照）
    * 测试站 i 的目标：s_i^(f) = mean_{j in J_i} [ t_ij - ref_j^(f) ]（i 本就不在 T_f，天然排除）
  即：训练与测试目标都由**同一个训练折参照**定义，donor pool 恒为训练折；差别仅在
  「本站观测是否已含在 donor pool 内」这一 LOO 约定。

输出：
  p4_cv_foldref.csv      各 blocking 的 out-of-fold R² 与条件增量（含 fold bootstrap CI）
  p4_cv_foldref_folds.csv 每折 ssr/sst，供复核与自助
  p4_perm_foldref.csv    999 次 block-preserving 置换零分布
  p4_cv_compare.csv      旧口径（全样本测试目标）vs 新口径（折内参照）逐项对照
"""
import os
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
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
m = m.reset_index(drop=True)

D = pd.read_csv(f"{T}/p0_station_event_tpk_multidef.csv", encoding="utf-8-sig")
D["st"] = D["st"].astype(int)
D = D[D["st"].isin(set(m["st_int"]))].reset_index(drop=True)

COL = "tpk_max"
BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
HYD = ["log_tt_sum", "log_up_area", "log_up_eff"]
FEATS = {
    "base": BASE,
    "hyd": BASE + HYD,
    "count": BASE + ["log_up"],
    "full": BASE + HYD + ["log_up"],
}

# ---------------------------------------------------------------------
# 0. 观测表（长表：st, ev_id, t_h）
# ---------------------------------------------------------------------
OBS = D[["st", "ev_id", COL]].dropna(subset=[COL]).copy()
OBS["t_h"] = OBS[COL].values * 0.25
# 每个事件的全局 sum / count（用于折内快速扣减）
EV_SUM = OBS.groupby("ev_id")["t_h"].sum()
EV_CNT = OBS.groupby("ev_id")["t_h"].size()


def fold_outcome(train_sts):
    """给定训练站集合，返回 pd.Series(index=st, value=s_i^(f))，覆盖全部 307 站。
    训练站用 LOO 参照（扣除自身），非训练站用完整训练折参照。"""
    is_tr = OBS["st"].isin(train_sts).values
    sub = OBS[is_tr]
    s_sum = sub.groupby("ev_id")["t_h"].sum()
    n_cnt = sub.groupby("ev_id")["t_h"].size()
    ref = (s_sum / n_cnt).rename("ref")           # 训练折事件参照

    g = OBS.merge(ref, left_on="ev_id", right_index=True, how="left")
    g = g.merge(n_cnt.rename("n_tr"), left_on="ev_id", right_index=True, how="left")
    g = g[g["ref"].notna() & (g["n_tr"] >= 1)].copy()

    own = g["st"].isin(train_sts).values
    # 训练站：ref_loo = (n*ref - t)/(n-1)，需 n>=2
    ok_tr = own & (g["n_tr"].values >= 2)
    resid = np.where(ok_tr,
                     g["t_h"].values - (g["n_tr"].values * g["ref"].values - g["t_h"].values)
                     / (g["n_tr"].values - 1),
                     g["t_h"].values - g["ref"].values)
    g["resid"] = resid
    g = g[ok_tr | (~own)]                          # 训练站 n<2 的事件剔除
    return g.groupby("st")["resid"].mean()


# 自检：全样本折 = 原始 Y（LOO 全样本）
Y_CHECK = fold_outcome(set(m["st_int"]))
diff = (m["st_int"].map(Y_CHECK) - m["se_tpk_max"]).abs().max()
print(f"[自检] fold_outcome(全体站) 与 master.se_tpk_max 最大绝对差 = {diff:.3e}")


# ---------------------------------------------------------------------
# 1. 折准备
# ---------------------------------------------------------------------
def prepare_folds(block_col):
    folds = []
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
        folds.append(dict(block=b, tr=tr_idx[ok], te=te_idx, ytr=ytr[ok], yte=yte))
    return folds


def evaluate(folds, df):
    """返回每折 ssr（各特征集）与 sst，以及合并 R²。"""
    recs = []
    for f in folds:
        tr, te = df.loc[f["tr"]], df.loc[f["te"]]
        ytr, yte = f["ytr"], f["yte"]
        good = ~np.isnan(yte)
        if good.sum() < 1:
            continue
        te, yte = te[good], yte[good]
        rec = {"block": f["block"], "n_te": len(te), "n_tr": len(tr),
               "sst": float(((yte - yte.mean()) ** 2).sum()),
               "sst_glob": float(((yte - 0.0) ** 2).sum())}
        # 全局中心化用：以全样本 out-of-fold 均值为中心（下方统一处理）
        rec["y_mean"] = float(yte.mean())
        rec["sum_y"] = float(yte.sum())
        rec["sum_y2"] = float((yte ** 2).sum())
        for name, feats in FEATS.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in feats])
            b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in feats])
            p = Xte @ b
            rec[f"ssr_{name}"] = float(((yte - p) ** 2).sum())
            rec[f"pred_{name}"] = p
            rec[f"y"] = yte
        recs.append(rec)
    return recs


def summarize(recs, tag):
    N = sum(r["n_te"] for r in recs)
    sy = sum(r["sum_y"] for r in recs)
    sy2 = sum(r["sum_y2"] for r in recs)
    sst_global = sy2 - sy ** 2 / N
    # 折内中心化只在 n_te>=2 的折上定义
    big = [r for r in recs if r["n_te"] >= 2]
    sst_w = np.array([r["sst"] for r in big])
    out = {"tag": tag, "n_folds": len(recs), "n": N,
           "n_folds_ge2": len(big), "n_ge2": int(sum(r["n_te"] for r in big))}
    for name in FEATS:
        ssr = np.array([r[f"ssr_{name}"] for r in recs])
        out[f"R2_{name}"] = 1 - ssr.sum() / sst_global
        ssr_w = np.array([r[f"ssr_{name}"] for r in big])
        out[f"R2w_{name}"] = 1 - ssr_w.sum() / sst_w.sum()
        out[f"ssr_{name}"] = ssr
        out[f"ssr_w_{name}"] = ssr_w
    out["sst"] = np.array([r["sst"] for r in recs])
    out["sst_w"] = sst_w
    out["sst_global"] = sst_global
    out["d_count_over_hyd"] = out["R2_full"] - out["R2_hyd"]
    out["d_hyd_over_count"] = out["R2_full"] - out["R2_count"]
    return out


def boot_delta(o, a="full", b="hyd", B=4000, seed=1):
    """按折重抽样的加权 ΔR² 置信区间（increment = (ssr_b - ssr_a)/Σsst）。
    sst=0 的孤立折不参与（其对 SST 无贡献，但保留在点估计中）。"""
    keep = o["sst"] > 0
    sst = o["sst"][keep]
    ssr_a = o[f"ssr_{a}"][keep]
    ssr_b = o[f"ssr_{b}"][keep]
    delta_f = (ssr_b - ssr_a) / sst
    w = sst / sst.sum()
    point = float((w * delta_f).sum())
    rng = np.random.default_rng(seed)
    G = len(delta_f)
    draws = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, G, G)
        draw = sst[idx]
        draw = draw if draw.sum() > 0 else np.ones(G)
        draws[i] = float((draw * delta_f[idx]).sum() / draw.sum())
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), float((draws > 0).mean())


def report(o):
    print(f"{o['tag']:24s} base={o['R2_base']:+7.4f} hyd={o['R2_hyd']:+7.4f} "
          f"count={o['R2_count']:+7.4f} full={o['R2_full']:+7.4f} | "
          f"Δc|h={o['d_count_over_hyd']:+7.4f} Δh|c={o['d_hyd_over_count']:+7.4f} "
          f"[折内中心化 R²: count={o['R2w_count']:+7.4f} full={o['R2w_full']:+7.4f}]")
    p, lo, hi, frac = boot_delta(o)
    print(f"{'':24s} Δc|h 折 bootstrap: {p:+.4f} [{lo:+.4f}, {hi:+.4f}], P(Δ>0)={frac:.3f}  "
          f"(折数 {o['n_folds']}, n={o['n']})")
    o["ci_dch"] = (p, lo, hi, frac)
    p2, lo2, hi2, frac2 = boot_delta(o, a="full", b="count")
    o["ci_dhc"] = (p2, lo2, hi2, frac2)
    print(f"{'':24s} Δh|c 折 bootstrap: {p2:+.4f} [{lo2:+.4f}, {hi2:+.4f}], P(Δ>0)={frac2:.3f}")
    return o


# ---------------------------------------------------------------------
# 2. 主结果
# ---------------------------------------------------------------------
print("\n" + "=" * 108)
print("2. 折内参照目标（训练与测试同参照）下的空间分块 CV")
print("=" * 108)

# 节点级 kmeans（与上一版一致，站点继承节点折标签）
node_coords = m.groupby("node_id")[["x", "y"]].mean()
for kk in [27, 10]:
    _, lab = kmeans2(node_coords.values.astype(float), kk, minit="points", seed=0)
    m[f"sp_k{kk}_node"] = m["node_id"].map(pd.Series(lab, index=node_coords.index)).values
    assert m.groupby("node_id")[f"sp_k{kk}_node"].nunique().max() == 1

BLOCKS = [("cl_max", "maximal upstream sets (26)"),
          ("cl_seg", "trunk segments (154)"),
          ("sp_k27_node", "spatial k-means k=27 (node)"),
          ("sp_k10_node", "spatial k-means k=10 (node)")]

print("\n" + "-" * 108)
print("分块大小分布（审稿 R3/R6 需要的簇结构）")
print("-" * 108)
for col, lab in BLOCKS:
    s = m.groupby(col).size()
    print(f"  {lab:34s} 块数={len(s):4d}  最小={s.min():4d} 中位={int(s.median()):4d} "
          f"最大={s.max():4d}  单站块={int((s==1).sum()):3d} "
          f"最大块占比={s.max()/len(m)*100:5.1f}%")

res = {}
rows = []
all_recs = {}
for col, lab in BLOCKS:
    fs = prepare_folds(col)
    rc = evaluate(fs, m)
    o = summarize(rc, lab)
    o = report(o)
    res[col] = o
    all_recs[col] = rc
    # 最大折单独贡献
    big = max(rc, key=lambda r: r["n_te"])
    r2b = {k: 1 - big[f"ssr_{k}"] / big["sst"] for k in FEATS}
    print(f"{'':24s} 主导折（n_te={big['n_te']}）: count={r2b['count']:+.4f} hyd={r2b['hyd']:+.4f} "
          f"full={r2b['full']:+.4f}  Δc|h={r2b['full']-r2b['hyd']:+.4f}  "
          f"该折占总 SST {big['sst']/sum(r['sst'] for r in rc)*100:.0f}%")
    rows.append({"blocking": lab, "n_folds": o["n_folds"], "n": o["n"],
                 "n_folds_ge2": o["n_folds_ge2"], "n_ge2": o["n_ge2"],
                 "largest_fold_n": big["n_te"],
                 "largest_fold_sst_share": big["sst"] / sum(r["sst"] for r in rc),
                 **{f"R2_{k}": o[f"R2_{k}"] for k in FEATS},
                 **{f"R2w_{k}": o[f"R2w_{k}"] for k in FEATS},
                 "d_count_over_hyd": o["d_count_over_hyd"],
                 "d_hyd_over_count": o["d_hyd_over_count"],
                 "ci_dch_lo": o["ci_dch"][1], "ci_dch_hi": o["ci_dch"][2],
                 "p_dch_positive": o["ci_dch"][3],
                 "ci_dhc_lo": o["ci_dhc"][1], "ci_dhc_hi": o["ci_dhc"][2],
                 "p_dhc_positive": o["ci_dhc"][3]})
df_cv = pd.DataFrame(rows)
df_cv.to_csv(f"{T}/p4_cv_foldref.csv", index=False, encoding="utf-8-sig")

frows = []
for col, lab in BLOCKS:
    for r in all_recs[col]:
        frows.append({"blocking": lab, "fold": r["block"], "n_te": r["n_te"],
                      "sst": r["sst"], "sum_y": r["sum_y"], "sum_y2": r["sum_y2"],
                      **{f"ssr_{k}": r[f"ssr_{k}"] for k in FEATS},
                      **{f"d_{k}": (r["ssr_hyd"] - r[f"ssr_{k}"]) / r["sst"] if r["sst"] > 0 else np.nan
                         for k in FEATS}})
pd.DataFrame(frows).to_csv(f"{T}/p4_cv_foldref_folds.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------
# 3. 置换检验（折内参照目标，999 次）
# ---------------------------------------------------------------------
print("\n" + "=" * 108)
print("3. block-preserving 置换（块内置换 log_up，999 次，折内参照目标）")
print("=" * 108)

base_logup = m["log_up"].to_numpy(float).copy()
NPERM = 999


def perm_test(block_col, label):
    fs = prepare_folds(block_col)
    obs = summarize(evaluate(fs, m), "obs")
    obs_d, obs_dh = obs["d_count_over_hyd"], obs["d_hyd_over_count"]
    print(f"\n  [{label}] 观测 Δc|h = {obs_d:+.4f},  Δh|c = {obs_dh:+.4f}；"
          f"R²count={obs['R2_count']:+.4f} R²hyd={obs['R2_hyd']:+.4f}")
    blocks_arr = m[block_col].to_numpy()
    rng = np.random.default_rng(20261102)
    pd_, pdh = np.empty(NPERM), np.empty(NPERM)
    for it in range(NPERM):
        lp = base_logup.copy()
        for b in np.unique(blocks_arr):
            idx = np.where(blocks_arr == b)[0]
            if len(idx) > 1:
                lp[idx] = rng.permutation(lp[idx])
        mm = m.copy()
        mm["log_up"] = lp
        o = summarize(evaluate(fs, mm), "perm")
        pd_[it] = o["d_count_over_hyd"]
        pdh[it] = o["d_hyd_over_count"]
    p_c = (1 + np.sum(pd_ >= obs_d)) / (NPERM + 1)
    p_h = (1 + np.sum(pdh >= obs_dh)) / (NPERM + 1)
    print(f"    零分布 Δc|h: mean={pd_.mean():+.4f}, SD={pd_.std():.4f}, "
          f"P50={np.percentile(pd_,50):+.4f}, P95={np.percentile(pd_,95):+.4f}, "
          f"P97.5={np.percentile(pd_,97.5):+.4f}, max={pd_.max():+.4f}")
    print(f"    单侧 p = {p_c:.4f}（超额 {int(np.sum(pd_>=obs_d))}/{NPERM}；最小可达 {1/(NPERM+1):.4f}）")
    print(f"    反向 Δh|c 单侧 p = {p_h:.4f}（超额 {int(np.sum(pdh>=obs_dh))}/{NPERM}）")
    return pd_, pdh, {"block": label, "obs_dch": obs_d, "p_dch": p_c,
                      "obs_dhc": obs_dh, "p_dhc": p_h,
                      "null_mean": pd_.mean(), "null_p975": np.percentile(pd_, 97.5)}


perm_rows = []
p1, p1h, s1 = perm_test("cl_max", "maximal upstream sets (26)")
perm_rows.append(s1)
p2, p2h, s2 = perm_test("sp_k27_node", "spatial k-means k=27 (node)")
perm_rows.append(s2)
pd.DataFrame({"perm_dch_clmax": p1, "perm_dhc_clmax": p1h,
              "perm_dch_k27": p2, "perm_dhc_k27": p2h}).to_csv(
    f"{T}/p4_perm_foldref.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(perm_rows).to_csv(f"{T}/p4_perm_foldref_summary.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------
# 4. 与旧口径对照
# ---------------------------------------------------------------------
print("\n" + "=" * 108)
print("4. 旧口径（测试目标=全样本 LOO 参照）在本脚本下的复现值")
print("=" * 108)


def evaluate_oldtarget(folds, df):
    """测试目标改用全样本 Y（= master.se_tpk_max），训练目标仍折内重建。"""
    YF = m.set_index("st_int")["se_tpk_max"]
    recs = []
    for f in folds:
        tr, te = df.loc[f["tr"]], df.loc[f["te"]]
        ytr = f["ytr"]
        yte = m.loc[f["te"], "st_int"].map(YF).to_numpy(float)
        good = ~np.isnan(yte)
        te, yte = te[good], yte[good]
        rec = {"n_te": len(te), "sst": float(((yte - yte.mean()) ** 2).sum()),
               "sum_y": float(yte.sum()), "sum_y2": float((yte ** 2).sum())}
        for name, feats in FEATS.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in feats])
            b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in feats])
            rec[f"ssr_{name}"] = float(((yte - Xte @ b) ** 2).sum())
        recs.append(rec)
    return recs


for col, lab in BLOCKS:
    fs = prepare_folds(col)
    o = summarize(evaluate_oldtarget(fs, m), lab)
    print(f"OLD {lab:30s} count={o['R2_count']:+7.4f} hyd={o['R2_hyd']:+7.4f} "
          f"Δc|h={o['d_count_over_hyd']:+7.4f} Δh|c={o['d_hyd_over_count']:+7.4f}")

print(f"\n总用时 {time.time()-t0:.1f}s")
