# -*- coding: utf-8 -*-
"""P4-1b 分块 CV 增量的不确定性：留一折 jackknife + 仅计有效折的 fold bootstrap + 逐折符号"""
import os
import numpy as np
import pandas as pd
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

T = os.environ.get("SEWER_TIMING_DATA",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "derived"))
f = pd.read_csv(f"{T}/p4_cv_foldref_folds.csv", encoding="utf-8-sig")
# 每个 blocking 的 R^2 分母：由汇总表的 R2_hyd 反推，保证与 Table 2 / 正文一致
s = pd.read_csv(f"{T}/p4_cv_foldref.csv", encoding="utf-8-sig").set_index("blocking")

# Table 2 panel D 三指标块一行的权威值（正文区间也以此为准）
T2 = {"maximal upstream sets (26)": (0.0243, -0.0357, 0.0842),
      "spatial k-means k=27 (node)": (0.0243, -0.0193, 0.0680),
      "spatial k-means k=10 (node)": (0.0255, -0.0162, 0.0673)}

rows = []
for lab, g in f.groupby("blocking", sort=False):
    g = g.reset_index(drop=True)
    ssr_h = g["ssr_hyd"].to_numpy()
    ssr_f = g["ssr_full"].to_numpy()
    ssr_c = g["ssr_count"].to_numpy()
    G = len(g)

    # 分母取该 blocking 计算 R^2 时实际使用的 SST，而不是逐折 sst 之和：
    # folds 表的 sst 是“折内均值”口径，单站折恒为 0，求和后与合并 SST 不等。
    SST = ssr_h.sum() / (1.0 - s.loc[lab, "R2_hyd"])

    def delta(a, b, D=SST):
        return (a.sum() - b.sum()) / D

    base_c = delta(ssr_h, ssr_f)      # count over hyd
    base_h = delta(ssr_c, ssr_f)      # hyd over count

    # 留一折 jackknife（分母固定，只减该折的分子贡献）
    jk_c = np.array([(ssr_h.sum() - ssr_h[i] - ssr_f.sum() + ssr_f[i]) / SST for i in range(G)])
    jk_h = np.array([(ssr_c.sum() - ssr_c[i] - ssr_f.sum() + ssr_f[i]) / SST for i in range(G)])
    se_c = np.sqrt((G - 1) / G * ((jk_c - jk_c.mean()) ** 2).sum())
    se_h = np.sqrt((G - 1) / G * ((jk_h - jk_h.mean()) ** 2).sum())

    # 逐折增量（全部折，含单站折；与 Table 2 注 (iii) 的符号计数同口径）
    per_c = (ssr_h - ssr_f) / SST
    per_h = (ssr_c - ssr_f) / SST

    # 仅有效折（n_te>=5）的 fold bootstrap
    ok = g["n_te"].to_numpy() >= 5
    h2, f2, c2 = ssr_h[ok], ssr_f[ok], ssr_c[ok]
    A1, A2 = h2 - f2, c2 - f2
    rng = np.random.default_rng(7)
    B = 5000
    d1, d2 = np.full(B, np.nan), np.full(B, np.nan)
    if len(A1) >= 2:                      # 只剩一个可重抽折时分块内 bootstrap 无定义
        for i in range(B):
            idx = rng.integers(0, len(A1), len(A1))
            d1[i] = A1[idx].mean() * len(A1) / SST
            d2[i] = A2[idx].mean() * len(A2) / SST
    d1, d2 = d1[~np.isnan(d1)], d2[~np.isnan(d2)]
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    ppos = lambda x: float(np.mean(x > 0)) if len(x) else float("nan")

    print(f"\n--- {lab} ---")
    print(f"  折数 {G}，其中 n_te>=5 的 {int(ok.sum())} 折（覆盖 {int(g['n_te'][ok].sum())}/{int(g['n_te'].sum())} 站）")
    print(f"  Δcount|hyd 点估计 {base_c:+.4f}；jackknife SE {se_c:.4f} → 95% {base_c-1.96*se_c:+.4f}~{base_c+1.96*se_c:+.4f}"
          f"；留一折范围 {jk_c.min():+.4f}~{jk_c.max():+.4f}")
    print(f"  逐折 Δcount|hyd（全部 {G} 折）: 正 {int((per_c>0).sum())}/{len(per_c)}，中位 {np.median(per_c):+.4f}")
    print(f"  有效折 fold bootstrap 95%: {base_c:+.4f} [{q(d1,2.5):+.4f}, {q(d1,97.5):+.4f}]，P(>0)={ppos(d1):.3f}")
    print(f"  Δhyd|count 点估计 {base_h:+.4f}；jackknife SE {se_h:.4f} → 95% {base_h-1.96*se_h:+.4f}~{base_h+1.96*se_h:+.4f}")
    print(f"  逐折 Δhyd|count: 正 {int((per_h>0).sum())}/{len(per_h)}，中位 {np.median(per_h):+.4f}；"
          f"bootstrap 95% [{q(d2,2.5):+.4f}, {q(d2,97.5):+.4f}]")

    if lab in T2:
        t2 = T2[lab]
        ok_t2 = (abs(base_c - t2[0]) < 5e-4
                 and abs((base_c - 1.96 * se_c) - t2[1]) < 5e-4
                 and abs((base_c + 1.96 * se_c) - t2[2]) < 5e-4)
        print(f"  对齐 Table 2 panel D（{t2[0]:+.4f} [{t2[1]:+.4f}, {t2[2]:+.4f}]）: {'PASS' if ok_t2 else 'FAIL'}")

    rows.append({"blocking": lab, "n_folds": G, "n_folds_ge5": int(ok.sum()),
                 "d_ch": base_c, "jk_se_ch": se_c,
                 "jk_lo_ch": base_c - 1.96 * se_c, "jk_hi_ch": base_c + 1.96 * se_c,
                 "jk_min_ch": jk_c.min(), "jk_max_ch": jk_c.max(),
                 "n_pos_ch": int((per_c > 0).sum()), "n_eff_folds": int(len(per_c)),
                 "boot_lo_ch": q(d1, 2.5), "boot_hi_ch": q(d1, 97.5),
                 "boot_p_pos_ch": ppos(d1),
                 "d_hc": base_h, "jk_se_hc": se_h,
                 "n_pos_hc": int((per_h > 0).sum()),
                 "boot_lo_hc": q(d2, 2.5), "boot_hi_hc": q(d2, 97.5),
                 "boot_p_pos_hc": ppos(d2)})

pd.DataFrame(rows).to_csv(f"{T}/p4_cv_foldref_uncertainty.csv", index=False, encoding="utf-8-sig")
print("\n已保存 p4_cv_foldref_uncertainty.csv")
