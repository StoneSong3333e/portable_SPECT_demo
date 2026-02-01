#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template similarity / error analysis
You only need to edit the CONFIG section below.
"""

# ============================================================
# ======================= CONFIG =============================
# ============================================================

ROI_QUANTILE = 0.90   # 保留最亮的 (1 - ROI_QUANTILE) 像素
                      # 0.90 = top 10%
                      # 0.95 = top 5%
                      # 0.98 = top 2%

USE_LOG1P = False
USE_BLUR = False

# ============================================================
# ============================================================

import os, glob, re, argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------------- utils ----------------
def angle_from_name(path: str) -> int:
    m = re.search(r"angle_(\d+)\.(csv|npy)$", os.path.basename(path))
    if not m:
        raise ValueError(f"Bad filename: {path}")
    return int(m.group(1))


def load_dense_map_csv(path: str, H=64, W=64) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (H, W):
        raise ValueError(f"Shape mismatch in {path}: got {arr.shape}, expected {(H, W)}")
    return arr


def normalize_sum(img: np.ndarray, eps=1e-12) -> np.ndarray:
    s = float(img.sum())
    if s < eps:
        return img * 0.0
    return img / s


def cosine_sim(A: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(A, axis=1, keepdims=True) + eps
    An = A / norms
    return An @ An.T


def pearson_corr(A: np.ndarray) -> np.ndarray:
    A0 = A - A.mean(axis=1, keepdims=True)
    return cosine_sim(A0)


def box_blur_3x3(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    out = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:]
    ) / 9.0
    return out


def preprocess(img: np.ndarray) -> np.ndarray:
    x = img
    if USE_LOG1P:
        x = np.log1p(x)
    if USE_BLUR:
        x = box_blur_3x3(x)
    return x


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maps_glob",
        default=os.path.join("analysis", "figures", "map_csv", "true_counts_map_angle_*.csv"),
        help="Dense maps glob"
    )
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    ap.add_argument("--outdir", default=os.path.join("analysis", "figures"))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save_roi", action="store_true")

    args = ap.parse_args()

    paths = sorted(glob.glob(args.maps_glob))
    if not paths:
        raise FileNotFoundError(f"No maps matched: {args.maps_glob}")

    os.makedirs(args.outdir, exist_ok=True)

    angles = [angle_from_name(p) for p in paths]

    # load + normalize
    imgs = []
    for p in paths:
        img = load_dense_map_csv(p, H=args.H, W=args.W)
        imgs.append(normalize_sum(img))
    imgs = np.stack(imgs, axis=0)  # [N,H,W]

    # ---------------- ROI ----------------
    sum_img = imgs.sum(axis=0)

    q = ROI_QUANTILE
    thr = np.quantile(sum_img, q)
    roi = sum_img >= thr

    roi_n = int(roi.sum())
    if roi_n <= 0:
        raise RuntimeError("ROI empty. Lower ROI_QUANTILE (e.g., 0.80, 0.85).")

    if args.save_roi:
        plt.figure(figsize=(5, 4))
        plt.imshow(roi.astype(float), origin="lower", aspect="auto")
        plt.title(f"ROI mask (quantile={ROI_QUANTILE}, top={(1-ROI_QUANTILE)*100:.1f}%, pixels={roi_n})")
        plt.tight_layout()
        out_roi = os.path.join(args.outdir, "template_error_roi_mask.png")
        plt.savefig(out_roi, dpi=200)
        plt.close()
        print("Saved:", out_roi)

    # ---------------- features ----------------
    A = np.stack(
        [preprocess(img)[roi].reshape(-1) for img in imgs],
        axis=0
    )

    # ---------------- similarity ----------------
    if args.metric == "cosine":
        S = cosine_sim(A)
    else:
        S = pearson_corr(A)

    # nearest neighbors
    print("==== Template nearest neighbors ====")
    order = np.argsort(angles)
    angs = np.array(angles)[order]
    S2 = S[order][:, order]

    for i, ang in enumerate(angs):
        row = S2[i].copy()
        row[i] = -np.inf
        nn = np.argsort(row)[::-1][:args.topk]
        msg = [f"{angs[j]}({row[j]:.4f})" for j in nn]
        print(f"angle {ang:3d} -> {', '.join(msg)}")

    # ---------------- heatmap ----------------
    plt.figure(figsize=(9, 8))
    plt.imshow(S2, origin="lower", aspect="auto")
    plt.colorbar(label=f"{args.metric} similarity")
    plt.xticks(np.arange(len(angs)), angs, rotation=90)
    plt.yticks(np.arange(len(angs)), angs)
    plt.title(
        f"Template Error ({args.metric}) | q={ROI_QUANTILE} (top {(1-ROI_QUANTILE)*100:.1f}%) "
        f"| log1p={'on' if USE_LOG1P else 'off'} blur={'on' if USE_BLUR else 'off'}"
    )
    plt.tight_layout()
    out_png = os.path.join(args.outdir, f"template_error_{args.metric}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)
    print(f"[INFO] ROI pixels: {roi_n}/{args.H*args.W}")
    print(f"[INFO] log1p={'on' if USE_LOG1P else 'off'}, blur={'on' if USE_BLUR else 'off'}")


if __name__ == "__main__":
    main()