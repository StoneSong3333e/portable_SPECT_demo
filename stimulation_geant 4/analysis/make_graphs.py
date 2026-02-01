#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 写死参数（按你要求）
# ======================
NY, NZ = 64,64
PHOTOPEAK_KEV = 140.0
WINDOW_PCT = 0.10  # ±10%
EMIN = PHOTOPEAK_KEV * (1.0 - WINDOW_PCT)
EMAX = PHOTOPEAK_KEV * (1.0 + WINDOW_PCT)

# 输入：analysis/graphs 下的 event_summary_angle_*.csv
INPUT_GLOB = os.path.join("analysis", "graphs", "event_summary_angle_*.csv")


OUTDIR_PNG = os.path.join("analysis", "figures", "map")
OUTDIR_CSV = os.path.join("analysis", "figures", "map_csv")

os.makedirs(OUTDIR_PNG, exist_ok=True)
os.makedirs(OUTDIR_CSV, exist_ok=True)


def parse_angle_tag(path: str) -> str:
    m = re.search(r"angle_(\d+)\.csv$", os.path.basename(path))
    return m.group(1) if m else "unknown"


def build_counts_map(df: pd.DataFrame) -> np.ndarray:
    """
    event_summary 每行一个 event，至少要有:
      - E_total_keV
      - iy
      - iz
    """
    required = {"E_total_keV", "iy", "iz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {df.columns.tolist()}")

    # 能窗筛选
    kept = df[(df["E_total_keV"] >= EMIN) & (df["E_total_keV"] <= EMAX)].copy()

    # 索引转 int + 范围裁剪（安全）
    kept["iy"] = kept["iy"].astype(int)
    kept["iz"] = kept["iz"].astype(int)
    kept = kept[(kept["iy"] >= 0) & (kept["iy"] < NY) & (kept["iz"] >= 0) & (kept["iz"] < NZ)]

    counts = np.zeros((NY, NZ), dtype=np.int64)
    np.add.at(counts, (kept["iy"].to_numpy(), kept["iz"].to_numpy()), 1)
    return counts


def save_dense_csv(counts: np.ndarray, out_csv: str):
    # 真正的 2D map（237x238）直接存下来，方便后面 template match
    np.savetxt(out_csv, counts, fmt="%d", delimiter=",")


def save_png(counts: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(counts, origin="lower", aspect="auto")
    plt.colorbar(label="Counts")
    plt.title(title)
    plt.xlabel("iz")
    plt.ylabel("iy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No files found: {INPUT_GLOB}\n"
            f"请确认 event_summary_angle_*.csv 在 B1_test/analysis/graphs/ 里。"
        )

    print(f"[INFO] Using energy window: {EMIN:.1f} – {EMAX:.1f} keV (140 ±10%)")
    print(f"[INFO] Found {len(paths)} event_summary files.")

    for p in paths:
        tag = parse_angle_tag(p)
        df = pd.read_csv(p)

        counts = build_counts_map(df)

        out_csv = os.path.join(OUTDIR_CSV, f"true_counts_map_angle_{tag}.csv")
        out_png = os.path.join(OUTDIR_PNG, f"true_counts_map_angle_{tag}.png")

        save_dense_csv(counts, out_csv)
        save_png(counts, out_png, title=f"True Counts Map @ angle {int(tag)} deg (140keV ±10%)")

        kept_n = int(((df["E_total_keV"] >= EMIN) & (df["E_total_keV"] <= EMAX)).sum())
        print(f"[OK] angle {tag}: kept {kept_n}/{len(df)} events -> {out_csv} + {out_png}")




if __name__ == "__main__":
    main()