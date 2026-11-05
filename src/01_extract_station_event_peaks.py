# -*- coding: utf-8 -*-
"""P0-1: 独立重建 307 站 × 事件 的 time-to-peak (tpk) 表。

动机（审计盲点2/3 + 数据继承链问题）：
  旧版 st_effect(s_i) 从 m1b_pairs_events.csv（sew-flu 站对表）继承 tpk_sew，
  其"有效事件集"取决于河道配对站是否同事件有效——非随机缺失，且非独立提取。
  本脚本从 dataset/5_sewer_telemetry_data 原始 CSV 独立重建 307 站的全时段水位，
  逐站×事件提取 global-maximum tpk（多峰值定义在 P0-2 做）。

输出: .workbuddy/tmp/p0_station_event_tpk.csv  (长表: st, ev_id, tpk_globalmax, amp_globalmax, cov, n_missing)
"""
import pandas as pd
import numpy as np
import os
import time
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.environ.get("SEWER_DATASET_ROOT", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir, os.pardir))
T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
t0 = time.time()

# ---- 1. 站名单（307 站）----
st_list = pd.read_csv(f"{T}/m3f_station_upstream.csv", encoding="utf-8-sig")
st_list["st"] = st_list["st"].astype(int)
print(f"站名单 n={len(st_list)}, st 范围 [{st_list.st.min()}, {st_list.st.max()}]")

# ---- 2. 站绑定表 ----
B = pd.read_csv(f"{T}/sensor_binding.csv", encoding="utf-8-sig")
print(f"sensor_binding 行数 {len(B)}")

# ---- 3. 索引表 ----
idx = pd.read_csv(f"{ROOT}/dataset/index_telemetry_data.csv", encoding="utf-8-sig", dtype=str)

# ---- 4. 事件表 ----
ev = pd.read_csv(f"{T}/m1a_events.csv", encoding="utf-8-sig")
print(f"事件数 {len(ev)}")

# ---- 5. 时间网格 ----
grid = pd.date_range("2024-02-19", "2026-05-01 00:00", freq="15min", tz="UTC")
gpos = {ts: i for i, ts in enumerate(grid)}
NG = len(grid)
print(f"时间网格 {NG} 步  {grid[0]} -> {grid[-1]}")

# ---- 6. 映射 st -> name/gtype/node_id/dataPath ----
# st 是 sensor_binding 的行索引（iloc）
rows = []
for sti in st_list["st"].values:
    if sti < 0 or sti >= len(B):
        rows.append((sti, None, None, None, None, None))
        continue
    brow = B.iloc[sti]
    name = str(brow["name"]).strip()
    gtype = str(brow["gtype"]).strip()
    node_id = str(brow["node_id"]).strip() if pd.notna(brow["node_id"]) else ""
    system = str(brow["system"]).strip() if pd.notna(brow["system"]) else ""
    # 找 dataPath
    m = idx[(idx["locationName"].str.strip() == name) & (idx["gaugeType"] == gtype)]
    dp = m["dataPath"].iloc[0] if len(m) > 0 else None
    rows.append((sti, name, gtype, node_id, system, dp))

map_df = pd.DataFrame(rows, columns=["st", "name", "gtype", "node_id", "system", "dataPath"])
miss = map_df["dataPath"].isna().sum()
print(f"映射完成: 缺 dataPath {miss}/{len(map_df)} 站")
if miss > 0:
    print("缺失站样例:")
    print(map_df[map_df.dataPath.isna()].head().to_string())

# ---- 7. 事件窗口参数（复用论文 §2.2）----
L = 512          # 128h 响应窗
WIN_PRE = 48     # 12h = 48 步
BASE_A = 96 * 4  # 96h
BASE_B = 24 * 4  # 24h
s_arr = ev["s"].values.astype(int)
e_arr = ev["e"].values.astype(int)
n_ev = len(ev)

# ---- 8. 逐站加载 + 提取 ----
def load_station(dp):
    """加载某站全时段 15min 水位(mm)，对齐 grid。"""
    p = os.path.join(ROOT, "dataset", dp)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, encoding="utf-8-sig")
    if "timestamp" not in d.columns:
        return None
    ts = pd.to_datetime(d["timestamp"], format="mixed", utc=True)
    cols = [c for c in d.columns if ("Level" in c) or ("Depth" in c)]
    if not cols:
        return None
    v = pd.to_numeric(d[cols[0]], errors="coerce")
    s = pd.Series(v.values, index=ts)
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("15min").mean()
    out = np.full(NG, np.nan)
    m = s.index.isin(grid)
    pos = [gpos[t] for t in s.index[m]]
    out[pos] = s.values[m]
    return out

records = []
for k, r in map_df.iterrows():
    sti = r["st"]
    dp = r["dataPath"]
    if dp is None:
        continue
    X = load_station(dp)
    if X is None:
        continue
    # 逐事件提取
    for ei in range(n_ev):
        s0 = s_arr[ei]
        b0, b1 = s0 - BASE_A, s0 - BASE_B
        w0 = s0 - WIN_PRE
        if b0 < 0 or w0 < 0 or w0 + L > NG:
            continue
        base_seg = X[b0:b1]
        if np.isnan(base_seg).all():
            continue
        bl = np.nanmedian(base_seg)
        seg = X[w0:w0 + L] - bl
        n_miss = int(np.isnan(seg).sum())
        if n_miss > 0.6 * L:
            continue
        cov = 1.0 - n_miss / L
        amp = np.nanmax(seg)          # global max 幅度
        tpk = np.nanargmax(seg)       # global max lag（步）
        records.append((sti, ei, tpk, amp, cov, n_miss))
    if (k + 1) % 50 == 0:
        print(f"  已处理 {k+1}/{len(map_df)} 站  [{time.time()-t0:.0f}s]")

D = pd.DataFrame(records, columns=["st", "ev_id", "tpk", "amp", "cov", "n_missing"])
print(f"\nstation×event 观测 {len(D)} 条  覆盖站 {D.st.nunique()}  事件 {D.ev_id.nunique()}")

# 每站有效事件数分布
cnt = D.groupby("st").size()
print(f"\n每站有效事件数: 中位 {cnt.median():.0f}  范围 [{cnt.min()}, {cnt.max()}]")
print(f"事件数 <10 的站: {(cnt < 10).sum()} 个")

out = f"{T}/p0_station_event_tpk.csv"
D.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n已存 {out}  耗时 {time.time()-t0:.1f}s")
