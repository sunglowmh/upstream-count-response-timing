# -*- coding: utf-8 -*-
"""P11-1 竞争规模变量：上游 link 数、上游总管长 与 count 的同折比较
=====================================================================
回应评审 #10：计数是否只是「上游 link 数」或「上游总管长」的替代变量。

口径：与 04_block_cv_fold_consistent.py 完全一致——同一分块、同一折、同一折内参照目标
（训练站 self-excluded、留出站用完整训练折均值）。自检两项：
  1) fold_outcome(全体站) 与 master.se_tpk_max 的最大绝对差应 <1e-12；
  2) 本脚本的 count/hyd 复现值应与 Table 2 面板 C 逐位一致。

变量定义（均在 flow-directed 图 G 上，与 up_size 同源）：
  up_size          严格上游节点数（论文的 count，表 2）
  n_up_link        上游子网内的有向边数（Conduit 与非 Conduit 都计）
  up_len_km        上游子网内 Conduit 边长之和（图边 length，几何长度，km）

输出 derived/p11_competing_vars.csv
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
    """给定训练站集合，返回 pd.Series(index=st, value=s_i^(f))，覆盖全部站。"""
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


diff = (m["st_int"].map(fold_outcome(set(m["st_int"]))) - m["se_tpk_max"]).abs().max()
print(f"[自检] fold_outcome(全体站) 与 se_tpk_max 最大绝对差 = {diff:.3e}")

# 节点级 kmeans（与 04 一致：kmeans2, minit="points", seed=0）
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
    """池化 out-of-fold R²：1 - ΣSSE_f / SST_pooled，与 04 的定义一致。"""
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
    return 1 - (e @ e).sum() / ((y - y.mean()) ** 2).sum()


# ---------------------------------------------------------------------
# 1. 两个竞争变量（与 up_size 同源，同一图）
# ---------------------------------------------------------------------
G = pickle.load(open(f"{PY}/.workbuddy/tmp/multi_rel_graph.gpickle", "rb"))
len_map = {}
for u, v, d in G.edges(data=True):
    if d.get("link_type") == "Conduit":
        lg = d.get("length")
        if lg is None or (isinstance(lg, float) and np.isnan(lg)):
            continue
        len_map[(u, v)] = float(lg)

n_link, up_len = {}, {}
for s in m.node_id:
    if s not in G:
        n_link[s], up_len[s] = 0, 0.0
        continue
    anc = nx.ancestors(G, s)
    anc.add(s)
    pairs = [(u, v) for u, v, _ in G.edges(data=True) if u in anc and v in anc and u != s]
    n_link[s] = len(pairs)
    up_len[s] = float(sum(len_map.get(e, 0.0) for e in pairs))

m["n_up_link"] = m.node_id.map(n_link).astype(float)
m["up_len_km"] = m.node_id.map(up_len).astype(float) / 1000.0
m["log_up_link"] = np.log1p(m["n_up_link"])
m["log_up_len"] = np.log1p(m["up_len_km"])

print("\n=== 1. 与 count 的关系 ===")
print("up_link 与 up_size 完全相等: %d / %d 站" % ((m.n_up_link == m.up_size).sum(), len(m)))
for c, lab in [("log_up_link", "link count"), ("log_up_len", "total length")]:
    print("  %-13s Pearson r=%+.5f  Spearman rho=%+.5f" % (
        lab, np.corrcoef(m.log_up, m[c])[0, 1],
        pd.Series(m.log_up).corr(pd.Series(m[c]), method="spearman")))

# ---------------------------------------------------------------------
# 2. 同折同目标的 OOF 比较
# ---------------------------------------------------------------------
print("\n=== 2. 同分块同目标 OOF R² ===")
print("%-34s %7s %8s %8s %8s %8s" % ("specification", "inR2", "cl_max", "trunk", "k27", "k10"))
SPECS = [("count (log up_size)", ["log_up"]),
         ("upstream link count", ["log_up_link"]),
         ("upstream total conduit length", ["log_up_len"]),
         ("count + link count", ["log_up", "log_up_link"]),
         ("count + total conduit length", ["log_up", "log_up_len"])]
rows = []
for lab, fs in SPECS:
    rr = r2_by_block(m, BASE + fs)
    print("%-34s %7.4f %8.4f %8.4f %8.4f %8.4f" % (
        lab, in_r2(m, BASE + fs), rr["cl_max"], rr["trunk"], rr["k27"], rr["k10"]))
    rows.append(dict(specification=lab, in_r2=in_r2(m, BASE + fs), **rr))
pd.DataFrame(rows).to_csv(f"{T}/p11_competing_vars.csv", index=False, encoding="utf-8-sig")

print("\n=== 3. 条件增量（同一分块同一目标）===")
for lab, c in [("link count", "log_up_link"), ("total length", "log_up_len")]:
    a = r2_by_block(m, BASE + [c])
    b = r2_by_block(m, BASE + [c, "log_up"])
    c0 = r2_by_block(m, BASE + ["log_up"])
    print("%-14s count|comp %+.4f %+.4f %+.4f %+.4f | comp|count %+.4f %+.4f %+.4f %+.4f" % (
        lab,
        b["cl_max"] - a["cl_max"], b["trunk"] - a["trunk"],
        b["k27"] - a["k27"], b["k10"] - a["k10"],
        b["cl_max"] - c0["cl_max"], b["trunk"] - c0["trunk"],
        b["k27"] - c0["k27"], b["k10"] - c0["k10"]))

# 校验：与 Table 2 面板 C 的 count / Στ 行逐位比对
chk = r2_by_block(m, BASE + ["log_up"])
print("\n[校验] count 行应等于 Table 2 面板 C：%s" % 
      "  ".join(f"{k}={v:.4f}" for k, v in chk.items()))

print(f"\n总用时 {time.time()-t0:.1f}s")
