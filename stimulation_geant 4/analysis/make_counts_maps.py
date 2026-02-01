#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd


PHOTOPEAK_KEV = 140.0
WINDOW_FRAC = 0.10  # ±10%
EMIN_KEV = PHOTOPEAK_KEV * (1.0 - WINDOW_FRAC)  # 126
EMAX_KEV = PHOTOPEAK_KEV * (1.0 + WINDOW_FRAC)  # 154


def parse_angle_from_filename(path: str) -> str:
    m = re.search(r"angle_(\d+)\.csv$", os.path.basename(path))
    return m.group(1) if m else "unknown"


def make_counts_map(
    df: pd.DataFrame,
    ny: int = None,
    nz: int = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    required = {"eventID", "iy", "iz", "edep_keV"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.copy()
    df["eventID"] = df["eventID"].astype(np.int64)
    df["iy"] = df["iy"].astype(np.int64)
    df["iz"] = df["iz"].astype(np.int64)
    df["edep_keV"] = df["edep_keV"].astype(float)

    # IMPORTANT: 建议你 batch 时固定 ny/nz，避免不同角度尺寸不一致
    if ny is None:
        ny = int(df["iy"].max()) + 1
    if nz is None:
        nz = int(df["iz"].max()) + 1

    g = df.groupby("eventID", sort=False)

    e_total = g["edep_keV"].sum().rename("E_total_keV")
    idx = g["edep_keV"].idxmax()

    winners = df.loc[idx, ["eventID", "iy", "iz", "edep_keV"]].copy()
    winners = winners.rename(columns={"edep_keV": "E_winnerStep_keV"}).set_index("eventID")

    event_table = pd.concat([e_total, winners], axis=1).reset_index()

    # Hard-coded energy window: photopeak ± 10%
    keep = (event_table["E_total_keV"] >= EMIN_KEV) & (event_table["E_total_keV"] <= EMAX_KEV)
    kept = event_table.loc[keep].copy()

    counts = np.zeros((ny, nz), dtype=np.int64)
    good = (kept["iy"] >= 0) & (kept["iy"] < ny) & (kept["iz"] >= 0) & (kept["iz"] < nz)
    kept = kept.loc[good]

    np.add.at(counts, (kept["iy"].to_numpy(), kept["iz"].to_numpy()), 1)

    return counts, event_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Input file or glob, e.g. maps/pixel_map_angle_*.csv")
    ap.add_argument("--outdir", default="analysis/counts_maps", help="Output directory for counts_map")
    ap.add_argument("--events_outdir", default="analysis/graphs", help="Output directory for event_summary")
    ap.add_argument("--ny", type=int, default=None, help="Override Ny (iy dimension)")
    ap.add_argument("--nz", type=int, default=None, help="Override Nz (iz dimension)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.events_outdir, exist_ok=True)

    paths = sorted(glob.glob(args.input))
    if not paths and os.path.isfile(args.input):
        paths = [args.input]
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input}")

    for path in paths:
        df = pd.read_csv(path)
        counts, event_table = make_counts_map(df, ny=args.ny, nz=args.nz)

        tag = parse_angle_from_filename(path)
        out_counts = os.path.join(args.outdir, f"counts_map_angle_{tag}.csv")
        out_events = os.path.join(args.events_outdir, f"event_summary_angle_{tag}.csv")

        ys, zs = np.nonzero(counts)
        out_df = pd.DataFrame({"iy": ys, "iz": zs, "count": counts[ys, zs]})
        out_df.to_csv(out_counts, index=False)

        event_table.to_csv(out_events, index=False)

        kept_n = int(((event_table["E_total_keV"] >= EMIN_KEV) & (event_table["E_total_keV"] <= EMAX_KEV)).sum())
        print(f"[OK] {os.path.basename(path)} -> kept {kept_n}/{len(event_table)} events "
              f"in window [{EMIN_KEV:.1f},{EMAX_KEV:.1f}] keV. Wrote {out_counts} and {out_events}")

if __name__ == "__main__":
    main()