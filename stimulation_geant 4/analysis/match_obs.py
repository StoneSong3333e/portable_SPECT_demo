#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
match_obs.py (Route A)
- Load templates from analysis/figures/map_csv/true_counts_map_angle_*.csv  (dense 64x64)
- Load observations from analysis/figures/observations/true_counts_map_angle_*.csv (dense 64x64)
- Compute cosine similarity (optionally on ROI pixels)
- For each obs, estimate angle by:
    1) discrete best template
    2) soft top-K interpolation (softmax weights)
    3) local 3-point parabola peak fitting (optional), with safety checks
- Run multiple template strides (e.g. 5/10/20/40/45 deg) in one shot.
- Also generates template-template similarity heatmaps (template_error) per stride.

No CLI required. Just:
    python3 analysis/match_obs.py
"""

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User config (edit here)
# =========================

# Paths (hard-coded)
TEMPLATES_GLOB = os.path.join("analysis", "figures", "map_csv", "true_counts_map_angle_*.csv")
OBS_DIR        = os.path.join("analysis", "figures", "observations")
OUTDIR_ROOT    = os.path.join("analysis", "figures", "routeA_multiStep_meanROI")

# Map shape
H, W = 64, 64

# Multi-step template stride sweep (degrees)
STRIDES = [5, 10, 20, 40, 45]

# Matching config
TOPK = 3
ALPHA = 20.0
EXCLUDE_SELF = True

# ROI config
ROI_MODE = "mean"     # "mean" | "variance" | "none"
ROI_FRAC = 0.20       # top fraction of pixels to keep
WHITEN = False        # normalize each pixel across angles: (x-mu)/sigma
EPS = 1e-12

# Parabola config
USE_PARABOLA = True
PARABOLA_REQUIRE_PEAK = True   # if True: only fit if middle point is local maximum
PARABOLA_CLAMP = True          # clamp theta* to [min(theta), max(theta)] to avoid crazy extrapolation

# Template-error heatmap config
TEMPLATE_ERROR_Q = 0.90   # show top (1-q) fraction? here used only as a label, full heatmap still saved

# =========================
# Utilities
# =========================

def angle_from_name(path: str) -> int:
    m = re.search(r"angle_(\d+)\.csv$", os.path.basename(path))
    if not m:
        raise ValueError(f"Bad filename: {path}")
    return int(m.group(1))


def load_dense_csv(path: str, H=64, W=64) -> np.ndarray:
    a = np.loadtxt(path, delimiter=",").astype(np.float64)
    if a.shape != (H, W):
        raise ValueError(f"Shape mismatch: {path} got {a.shape}, expected {(H,W)}")
    return a


def normalize_sum(x: np.ndarray, eps=1e-12) -> np.ndarray:
    s = float(x.sum())
    return x * 0.0 if s < eps else x / s


def cosine_sim(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float((a @ b) / (na * nb))


def soft_angle(top_angles: np.ndarray, top_sims: np.ndarray, alpha: float) -> float:
    # stable softmax weights
    top_angles = np.asarray(top_angles, dtype=np.float64)
    top_sims = np.asarray(top_sims, dtype=np.float64)
    w = np.exp(alpha * (top_sims - np.max(top_sims)))
    return float(np.sum(w * top_angles) / (np.sum(w) + 1e-12))


def parabola_peak_from_three(theta: np.ndarray, sim: np.ndarray,
                             require_peak: bool = True,
                             clamp: bool = True) -> float:
    """
    Fit quadratic sim = a*theta^2 + b*theta + c using 3 points,
    return vertex theta* = -b/(2a).

    Safety:
    - if require_peak=True, only fit when middle point is local maximum:
        sim[1] >= sim[0] and sim[1] >= sim[2]
      otherwise return theta[argmax(sim)] (discrete)
    - if abs(a) ~ 0, fallback to discrete
    - if clamp=True, clamp theta* into [min(theta), max(theta)]
    """
    theta = np.asarray(theta, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)

    if theta.size != 3 or sim.size != 3:
        return float(theta[np.argmax(sim)])

    # enforce distinct
    if len(set(theta.tolist())) != 3:
        return float(theta[np.argmax(sim)])

    if require_peak:
        # only fit if middle is a peak
        if not (sim[1] >= sim[0] and sim[1] >= sim[2]):
            return float(theta[np.argmax(sim)])

    # quadratic fit
    a, b, c = np.polyfit(theta, sim, 2)
    if abs(a) < 1e-12:
        return float(theta[np.argmax(sim)])

    t_star = -b / (2.0 * a)

    if clamp:
        t_min, t_max = float(theta.min()), float(theta.max())
        if t_star < t_min:
            t_star = t_min
        if t_star > t_max:
            t_star = t_max

    return float(t_star)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# ROI building
# =========================

def build_roi_mask(templates_stack: np.ndarray, mode: str, frac: float) -> np.ndarray:
    """
    templates_stack: [Nt, H, W] already normalized per-map (sum=1)
    Returns mask [H, W] boolean.
    """
    mode = mode.lower().strip()
    if mode == "none":
        return np.ones((templates_stack.shape[1], templates_stack.shape[2]), dtype=bool)

    if mode == "mean":
        score = templates_stack.mean(axis=0)  # [H,W]
    elif mode == "variance":
        score = templates_stack.var(axis=0)   # [H,W]
    else:
        raise ValueError(f"Unknown ROI_MODE={mode}")

    flat = score.reshape(-1)
    k = int(np.ceil(frac * flat.size))
    k = max(1, min(k, flat.size))
    # top-k
    thresh = np.partition(flat, -k)[-k]
    mask = (score >= thresh)
    return mask


def maybe_whiten_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    vectors: [N, D]
    whiten across N for each pixel dimension:
        z = (x - mu) / sigma
    """
    mu = vectors.mean(axis=0, keepdims=True)
    sigma = vectors.std(axis=0, keepdims=True)
    return (vectors - mu) / (sigma + eps)


def save_roi_debug(templates_stack: np.ndarray, roi_mask: np.ndarray, outdir: str, mode: str):
    """
    Save mean/var maps and ROI mask as images.
    """
    ensure_dir(outdir)
    mean_map = templates_stack.mean(axis=0)
    var_map  = templates_stack.var(axis=0)

    # mean map
    plt.figure(figsize=(5.8, 5))
    plt.imshow(mean_map, origin="lower")
    plt.title("mean_map (templates)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_map.png"), dpi=200)
    plt.close()

    # var map
    plt.figure(figsize=(5.8, 5))
    plt.imshow(var_map, origin="lower")
    plt.title("var_map (templates)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "variance_map.png"), dpi=200)
    plt.close()

    # roi mask
    plt.figure(figsize=(5.8, 5))
    plt.imshow(roi_mask.astype(np.int32), origin="lower")
    plt.title(f"roi_mask ({mode}, top {ROI_FRAC*100:.1f}%)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roi_mask.png"), dpi=200)
    plt.close()


# =========================
# Template error heatmap
# =========================

def template_error_heatmap(tmpl_angles: np.ndarray, tmpl_vecs: np.ndarray, out_png: str, title: str):
    """
    Compute Nt x Nt cosine similarity matrix among templates and save heatmap.
    tmpl_vecs: [Nt, D]
    """
    Nt = tmpl_vecs.shape[0]
    M = np.zeros((Nt, Nt), dtype=np.float64)

    # Normalize each vec once
    norms = np.linalg.norm(tmpl_vecs, axis=1) + EPS
    vn = tmpl_vecs / norms[:, None]

    # cosine matrix
    M = vn @ vn.T

    plt.figure(figsize=(8.2, 7))
    plt.imshow(M, origin="lower", vmin=np.nanmin(M), vmax=1.0)
    plt.colorbar(label="cosine similarity")
    plt.title(title)
    plt.xlabel("template index")
    plt.ylabel("template index")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return M


# =========================
# Main routine
# =========================

def main():
    ensure_dir(OUTDIR_ROOT)

    # ---- load all templates (full set) ----
    tpaths_all = sorted(glob.glob(TEMPLATES_GLOB))
    if not tpaths_all:
        raise FileNotFoundError(f"No templates matched: {TEMPLATES_GLOB}")

    # load as stack for ROI computation (use full set for ROI)
    tmpl_angles_all = np.array([angle_from_name(p) for p in tpaths_all], dtype=int)
    tmpl_maps_all = np.stack([normalize_sum(load_dense_csv(p, H, W)) for p in tpaths_all], axis=0)  # [Nt,H,W]

    # ---- load observations (all files in OBS_DIR) ----
    opaths = sorted(glob.glob(os.path.join(OBS_DIR, "true_counts_map_angle_*.csv")))
    if not opaths:
        raise FileNotFoundError(f"No observation files in: {OBS_DIR}")

    obs_angles = np.array([angle_from_name(p) for p in opaths], dtype=int)

    print(f"[INFO] Templates(all): {len(tpaths_all)} files from {TEMPLATES_GLOB}")
    print(f"[INFO] Observations : {len(opaths)} files from {OBS_DIR}")
    print(f"[INFO] Config: STRIDES={STRIDES}, TOPK={TOPK}, ALPHA={ALPHA}, EXCLUDE_SELF={EXCLUDE_SELF}, USE_PARABOLA={USE_PARABOLA}")
    print(f"[INFO] ROI: mode={ROI_MODE}, top {ROI_FRAC*100:.1f}% pixels, WHITEN={WHITEN}")
    if USE_PARABOLA:
        print(f"[INFO] Parabola: REQUIRE_PEAK={PARABOLA_REQUIRE_PEAK}, CLAMP={PARABOLA_CLAMP}")

    # ---- build ROI mask from templates (full set) ----
    roi_mask = build_roi_mask(tmpl_maps_all, ROI_MODE, ROI_FRAC)
    roi_idx = np.flatnonzero(roi_mask.reshape(-1))
    print(f"[INFO] ROI pixels: {roi_idx.size}/{H*W}")

    # save ROI debug
    save_roi_debug(tmpl_maps_all, roi_mask, OUTDIR_ROOT, ROI_MODE)
    print(f"[OK] wrote ROI debug images to {OUTDIR_ROOT}")

    # ---- pre-load observation vectors (normalized) ----
    obs_vecs_full = []
    for p in opaths:
        m = normalize_sum(load_dense_csv(p, H, W)).reshape(-1)
        obs_vecs_full.append(m)
    obs_vecs_full = np.stack(obs_vecs_full, axis=0)  # [No, D]

    # apply ROI
    obs_vecs = obs_vecs_full[:, roi_idx]  # [No, Droi]

    # optional whiten (across templates+obs separately is tricky; simplest: whiten using templates stats)
    # We'll whiten per stride after selecting templates.
    # (obs will be whitened using template mu/sigma for that stride)

    # ---- run each stride ----
    for stride in STRIDES:
        # select templates near multiples of stride (and always include 0 and 180 if present)
        sel = (tmpl_angles_all % stride == 0)
        sel_idx = np.where(sel)[0]
        tpaths = [tpaths_all[i] for i in sel_idx]
        tmpl_angles = tmpl_angles_all[sel_idx]
        tmpl_maps = tmpl_maps_all[sel_idx]  # [Nt, H, W]

        # sort by angle
        order = np.argsort(tmpl_angles)
        tmpl_angles = tmpl_angles[order]
        tmpl_maps = tmpl_maps[order]

        Nt = tmpl_maps.shape[0]
        step_out = os.path.join(OUTDIR_ROOT, f"step{stride:02d}")
        ensure_dir(step_out)

        print("\n" + "="*70)
        print(f"[RUN] stride={stride} deg | templates={Nt} | out={step_out}")
        print("="*70)

        # build template vectors (ROI applied)
        tmpl_vecs_full = tmpl_maps.reshape(Nt, -1)  # [Nt, Dfull]
        tmpl_vecs = tmpl_vecs_full[:, roi_idx]      # [Nt, Droi]

        # optional whitening using templates stats
        if WHITEN:
            mu = tmpl_vecs.mean(axis=0, keepdims=True)
            sigma = tmpl_vecs.std(axis=0, keepdims=True) + EPS
            tmpl_vecs_use = (tmpl_vecs - mu) / sigma
            obs_vecs_use = (obs_vecs - mu) / sigma
        else:
            tmpl_vecs_use = tmpl_vecs
            obs_vecs_use = obs_vecs

        # ---- template error heatmap ----
        heat_png = os.path.join(step_out, f"template_error_cosine_step{stride:02d}.png")
        title = f"Template Error (cosine) | stride={stride} | ROI={ROI_MODE}({ROI_FRAC*100:.0f}%) | whiten={WHITEN}"
        _ = template_error_heatmap(tmpl_angles, tmpl_vecs_use, heat_png, title)
        print(f"[OK] wrote {heat_png}")

        # ---- match each observation ----
        rows = []
        abs_err_est = []
        abs_err_soft = []
        abs_err_para = []

        for i, (obs_path, theta_true) in enumerate(zip(opaths, obs_angles)):
            ov = obs_vecs_use[i]  # [Droi]

            # cosine to all templates
            sims = np.array([cosine_sim(tv, ov, EPS) for tv in tmpl_vecs_use], dtype=np.float64)
            sims_use = sims.copy()

            if EXCLUDE_SELF:
                same = (tmpl_angles == int(theta_true))
                sims_use[same] = -np.inf

            best_idx = int(np.argmax(sims_use))
            theta_est = int(tmpl_angles[best_idx])
            best_cos = float(sims[best_idx])

            # top-k
            k = min(TOPK, Nt)
            top_idx = np.argsort(sims_use)[::-1][:k]
            top_angles = tmpl_angles[top_idx].astype(np.float64)
            top_sims = sims[top_idx].astype(np.float64)
            theta_soft = soft_angle(top_angles, top_sims, ALPHA)

            # local parabola: choose 3 points centered around best angle in the *angle-sorted* template list
            theta_para = np.nan
            triplet_angles = None
            triplet_sims = None
            if USE_PARABOLA and Nt >= 3:
                # position in sorted array
                pos = int(np.where(tmpl_angles == theta_est)[0][0])
                # pick neighbors: pos-1,pos,pos+1 with clamping at edges
                if pos == 0:
                    idx3 = [0, 1, 2]
                elif pos == Nt - 1:
                    idx3 = [Nt - 3, Nt - 2, Nt - 1]
                else:
                    idx3 = [pos - 1, pos, pos + 1]

                triplet_angles = tmpl_angles[idx3].astype(np.float64)
                triplet_sims = sims[idx3].astype(np.float64)

                theta_para = parabola_peak_from_three(
                    triplet_angles, triplet_sims,
                    require_peak=PARABOLA_REQUIRE_PEAK,
                    clamp=PARABOLA_CLAMP
                )

            err_est = float(theta_est - theta_true)
            err_soft = float(theta_soft - theta_true)
            err_para = float(theta_para - theta_true) if np.isfinite(theta_para) else np.nan

            abs_err_est.append(abs(err_est))
            abs_err_soft.append(abs(err_soft))
            if np.isfinite(err_para):
                abs_err_para.append(abs(err_para))

            print("------------------------------------")
            print(f"OBS        : {os.path.basename(obs_path)}")
            print(f"theta_true : {theta_true} deg")
            print(f"theta_est  : {theta_est} deg   (best cosine={best_cos:.6f}) | err={err_est:+.0f} deg")
            print(f"theta_soft : {theta_soft:.2f} deg (TOPK={k}, ALPHA={ALPHA:g}) | err={err_soft:+.2f} deg")
            if USE_PARABOLA:
                print(f"theta_para : {theta_para:.2f} deg (local 3-pt) | err={err_para:+.2f} deg")
                print(f"   triplet : {triplet_angles.astype(int).tolist()} sims={triplet_sims.tolist()}")
            print(" top matches:")
            for r, j in enumerate(top_idx, 1):
                print(f"  {r:02d}) angle={tmpl_angles[j]:3d}  cosine={sims[j]:.6f}")

            rows.append({
                "obs_file": os.path.basename(obs_path),
                "theta_true": int(theta_true),
                "theta_est": int(theta_est),
                "theta_soft": float(theta_soft),
                "theta_para": float(theta_para) if np.isfinite(theta_para) else np.nan,
                "error_deg": float(err_est),
                "error_soft_deg": float(err_soft),
                "error_para_deg": float(err_para) if np.isfinite(err_para) else np.nan,
                "best_cosine": float(best_cos),
                "topk": int(k),
                "alpha": float(ALPHA),
                "stride": int(stride),
                "roi_mode": ROI_MODE,
                "roi_frac": float(ROI_FRAC),
                "whiten": bool(WHITEN),
                "parabola_require_peak": bool(PARABOLA_REQUIRE_PEAK),
            })

        # ---- save CSV ----
        df = pd.DataFrame(rows).sort_values("theta_true").reset_index(drop=True)
        out_csv = os.path.join(step_out, f"routeA_errors_step{stride:02d}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv}")

        # ---- plot error curves ----
        plt.figure(figsize=(7, 4.5))
        plt.plot(df["theta_true"], df["error_deg"], "o-", label="discrete (theta_est - true)")
        plt.plot(df["theta_true"], df["error_soft_deg"], "o-", label="soft (theta_soft - true)")
        if USE_PARABOLA:
            plt.plot(df["theta_true"], df["error_para_deg"], "o-", label="parabola (theta_para - true)")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("true angle (deg)")
        plt.ylabel("error (deg)")
        plt.title(f"Route A error vs angle (stride={stride}, ROI={ROI_MODE}, frac={ROI_FRAC})")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(step_out, f"routeA_error_curve_step{stride:02d}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] wrote {out_png}")

        # ---- summary ----
        mae_est = float(np.mean(abs_err_est)) if abs_err_est else np.nan
        mae_soft = float(np.mean(abs_err_soft)) if abs_err_soft else np.nan
        mae_para = float(np.mean(abs_err_para)) if abs_err_para else np.nan
        print(f"[SUMMARY stride={stride:02d}] MAE discrete={mae_est:.3f} deg | MAE soft={mae_soft:.3f} deg | MAE parabola={mae_para:.3f} deg")

    print("\n[DONE] All strides finished. Output folder:")
    print(f"  {OUTDIR_ROOT}")


if __name__ == "__main__":
    main()