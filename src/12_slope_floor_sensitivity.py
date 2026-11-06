# -*- coding: utf-8 -*-
"""P12-1 Στ 坡度下限敏感性：floor ∈ {1e-5,1e-4,5e-4,1e-3} 与「剔除非正坡度」
=====================================================================
回应评审 #5：13,443 条 Conduit 中 274 条（2.04%）坡度 ≤0，Στ 把它们统一 floor 到
1e-4 而非剔除。本脚本检验 count − Στ 的比较对这一处置是否敏感。

口径关键：Στ 必须在 m13o_travel_time.py 的逻辑内部改 EPS 重算，不能另写一套复算。
并行管（同一 (us,ds) 多条 Conduit）按最小 travel_time 去重，这一规则与 m13o 相同；
另写复算会因去重口径不同产生 ρ≈0.91 的偏差，不可入稿。
自检：floor=1e-4 的 Στ 与 master.up_travel_sum 的最大绝对差应 <1e-12。

输出 derived/p12_slope_floor.csv
"""
import os
import sys
import io
import time
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.vq import kmeans2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()

PY = os.environ.get("SEWER_TIMING_ROOT", "E:/work/workBuddy/BlueHeart")
T = os.environ.get("SEWER_TIMING_DATA",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
BASE = ["x_std", "y_std", "sys_foul", "sys_combined"]
N_MANNING = 0.013

m = pd.read_csv(f"{T}/p1_h3_master.csv", encoding="utf-8-sig")
for c in ["st", "node_id"]:
    m[c] = m[c].astype(str).str.strip()
m["st_int"] = m["st"].astype(int)
m = m.reset_index(drop=True)

D = pd.read_csv(f"{T}/p0_station_event_tpk_multidef.csv", encoding="utf-8-sig")
D["st"] = D["st"].astype(int)
D = D[D["st"].isin(set(m["st_int"]))].reset_index(drop=True)
OBS = D[["st", "ev_id", "tpk_max"]].dropna(subset=["tpk_max"]).copy()
OBS["t_h"] = OBS["tpk_max"].values * 0.25


def fold_outcome(train_sts):
    is_tr = OBS["st"].isin(train_sts).values
    sub = OBS[is_tr]
    ref = (sub.groupby("ev_id")["t_h"].sum() / sub.groupby("ev_id")["t_h"].size()).rename("ref")
    n_cnt = sub.groupby("ev_id")["t_h"].size().rename("n_tr")
    g = OBS.merge(ref, left_on="ev_id", right_index=True, how="left").merge(
        n_cnt, left_on="ev_id", right_index=True, how="left")
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


node_coords = m.groupby("node_id")[["x", "y"]].mean()
for kk in [27, 10]:
    _, lab = kmeans2(node_coords.values.astype(float), kk, minit="points", seed=0)
    m[f"sp_k{kk}_node"] = m["node_id"].map(pd.Series(lab, index=node_coords.index)).values

BLOCKS = [("cl_max", "maximal upstream sets (26)"),
          ("cl_seg", "trunk segments (154)"),
          ("sp_k27_node", "spatial k-means k=27"),
          ("sp_k10_node", "spatial k-means k=10")]
LAB = {"cl_max": "cl_max", "cl_seg": "trunk", "sp_k27_node": "k27", "sp_k10_node": "k10"}


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


FOLDS = {col: prepare_folds(col) for col, _ in BLOCKS}


def r2_by_block(df, feats):
    out = {}
    for col, _ in BLOCKS:
        pred = np.full(len(df), np.nan)
        y = np.full(len(df), np.nan)
        for f in FOLDS[col]:
            tr, te = df.loc[f["tr"]], df.loc[f["te"]]
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in feats])
            b, *_ = np.linalg.lstsq(Xtr, f["ytr"], rcond=None)
            Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in feats])
            pred[f["te"]] = Xte @ b
            y[f["te"]] = f["yte"]
        ok = np.isfinite(y) & np.isfinite(pred)
        out[LAB[col]] = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    return out


def in_r2(df, feats, ycol="se_tpk_max"):
    d = df[[ycol] + feats].dropna()
    y = d[ycol].to_numpy(float)
    X = np.column_stack([np.ones(len(y))] + [d[c].to_numpy(float) for c in feats])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - (e @ e) / ((y - y.mean()) ** 2).sum()

# ---------------------------------------------------------------------
# 1. 图边长度与每站上游边集（一次算好，各 EPS 复用）
# ---------------------------------------------------------------------
G = pickle.load(open(f"{PY}/.workbuddy/tmp/multi_rel_graph.gpickle", "rb"))
len_map = {}
for u, v, d in G.edges(data=True):
    if d.get("link_type") == "Conduit":
        lg = d.get("length")
        if lg is None or (isinstance(lg, float) and np.isnan(lg)):
            continue
        len_map[(u, v)] = float(lg)

up_pairs = {}
for s in m.node_id:
    if s not in G:
        up_pairs[s] = []
        continue
    anc = nx.ancestors(G, s)
    anc.add(s)
    up_pairs[s] = [(u, v) for u, v, _ in G.edges(data=True) if u in anc and v in anc and u != s]

L = pd.read_csv(f"{PY}/dataset/2_network_shape_files/links/links_metadata.csv")
L["us"] = L.us_node_id.astype(str).str.strip()
L["ds"] = L.ds_node_id.astype(str).str.strip()
CD = L[L.link_type == "Conduit"].copy()
CD["S_raw"] = pd.to_numeric(CD.gradient, errors="coerce")
CD["length"] = [len_map.get((a, b), np.nan) for a, b in zip(CD.us, CD.ds)]
print("Conduit 行 %d | 长度可用 %d | 坡度缺失 %d | 坡度≤0 %d（=0 %d, <0 %d）" % (
    len(CD), CD.length.notna().sum(), CD.S_raw.isna().sum(),
    (CD.S_raw <= 0).sum(), (CD.S_raw == 0).sum(), (CD.S_raw < 0).sum()))

CD["width_m"] = CD.conduit_width / 1000.0
K = (1.0 / N_MANNING) * ((CD.width_m / 4.0) ** (2.0 / 3.0))


def tau_for(S):
    """复刻 m13o_travel_time.py：并行管按最小 travel_time 去重。"""
    tt = (CD.length / (K * np.sqrt(S))) / 3600.0
    d = pd.DataFrame({"us": CD.us, "ds": CD.ds, "tt": tt}).dropna()
    d = d.sort_values("tt").drop_duplicates(subset=["us", "ds"], keep="first")
    return dict(zip(zip(d.us, d.ds), d.tt))

# ---------------------------------------------------------------------
# 2. 五个处置 × 同一折同一目标
# ---------------------------------------------------------------------
VARIANTS = [("floor 1e-5", CD.S_raw.clip(lower=1e-5)),
            ("floor 1e-4 (reported)", CD.S_raw.clip(lower=1e-4)),
            ("floor 5e-4", CD.S_raw.clip(lower=5e-4)),
            ("floor 1e-3", CD.S_raw.clip(lower=1e-3)),
            ("drop non-positive", CD.S_raw.where(CD.S_raw > 0))]

print("\n%-24s %7s %7s %7s %7s %7s %7s %7s" % (
    "variant", "medTau", "inR2", "inD", "cl_max", "trunk", "k27", "k10"))
rows = []
for lab, S in VARIANTS:
    tm = tau_for(S)
    tot = {s: float(sum(tm.get(e, 0.0) for e in p)) for s, p in up_pairs.items()}
    m["tau_v"] = m.node_id.map(tot).astype(float)
    m["l_tau_v"] = np.log1p(m["tau_v"])
    if "reported" in lab:
        dv = (m["tau_v"] - m["up_travel_sum"]).abs().max()
        print("[自检] 与 master.up_travel_sum 最大绝对差 = %.3e" % dv)
    a = r2_by_block(m, BASE + ["l_tau_v"])
    b = r2_by_block(m, BASE + ["l_tau_v", "log_up"])
    in_a = in_r2(m, BASE + ["l_tau_v"])
    in_b = in_r2(m, BASE + ["l_tau_v", "log_up"])
    print("%-24s %7.3f %7.4f %+7.4f %8.4f %8.4f %8.4f %8.4f" % (
        lab, m.tau_v.median(), in_a, in_b - in_a,
        a["cl_max"], a["trunk"], a["k27"], a["k10"]))
    print("%-24s %7s %7s %7s %+8.4f %+8.4f %+8.4f %+8.4f" % (
        "", "", "", "count|Stau",
        b["cl_max"] - a["cl_max"], b["trunk"] - a["trunk"],
        b["k27"] - a["k27"], b["k10"] - a["k10"]))
    rows.append(dict(variant=lab, med_tau_h=m.tau_v.median(), max_tau_h=m.tau_v.max(),
                     n_edge=len(tm), in_hyd=in_a, in_full=in_b,
                     in_dch=in_b - in_a,
                     oof_hyd_clmax=a["cl_max"], oof_hyd_trunk=a["trunk"],
                     oof_hyd_k27=a["k27"], oof_hyd_k10=a["k10"],
                     dch_clmax=b["cl_max"] - a["cl_max"],
                     dch_trunk=b["trunk"] - a["trunk"],
                     dch_k27=b["k27"] - a["k27"],
                     dch_k10=b["k10"] - a["k10"]))
pd.DataFrame(rows).to_csv(f"{T}/p12_slope_floor.csv", index=False, encoding="utf-8-sig")

print(f"\n总用时 {time.time()-t0:.1f}s")
