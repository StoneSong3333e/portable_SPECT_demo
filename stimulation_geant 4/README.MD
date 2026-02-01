# SPECT Angle Observability Demo  
*(Geant4 Simulation + Image-Based Analysis)*

## Overview

This folder contains a **minimal, fully reproducible Geant4-based simulation and analysis demo**
investigating **angular observability** in side-view SPECT projections.

The purpose of this demo is to answer a focused feasibility question:

> **Do raw photon-counting SPECT projections contain recoverable angular information,
even under coarse angular sampling and realistic counting noise?**

This work is designed as an **observability and feasibility study**, rather than a full
tomographic reconstruction or geometric calibration pipeline.

---

## Repository Structure

```text
simulation_geant4/
├── CMakeLists.txt
├── include/          # Geant4 C++ headers (geometry, source, detector, actions)
├── src/              # Geant4 C++ implementation
├── analysis/         # Python analysis: template matching, ROI, interpolation
└── data/
    ├── template/     # Template library (dense 2D count maps at known angles)
    ├── observations/ # Observation maps (unknown angles + noise)
    └── results/      # Similarity maps, ROI debug plots, error curves
---

## Simulation Model

The simulation uses a **custom-designed simplified 3-point activity phantom**,
rather than a standard clinical phantom, in order to isolate and emphasize
**pure angular information content** in projection data.

Key assumptions:

- The phantom rotates around a fixed axis (Z-axis)
- A virtual detector observes the phantom from a fixed side-view direction
- Each rotation angle produces a 2D photon-count map
- Photon transport and detection are handled by **Geant4**
- Finite-count Poisson noise and small angular perturbations are optionally applied

All projections are exported as dense 2D CSV count maps (e.g., 64×64),
enabling transparent and reproducible downstream analysis.

---

## Template Library vs Observations

### Template Library
- Generated at discrete rotation angles over 0–180°
- Angular step size configurable (e.g., 5°, 10°, 20°, 40°, 45°)
- Serves as a mapping: **angle → projection image**

### Observation Set
- A small set of projections at unknown angles
- Includes realistic effects:
  - Photon counting noise
  - Angular uncertainty
- Used to test whether angle can be inferred from image data alone

---

## Analysis Pipeline (Python)

The `analysis/` directory implements a lightweight but physically meaningful
image-based angle estimation pipeline.

### 1. Template Similarity Analysis

- Pairwise cosine similarity is computed between template projections
- Produces **template similarity / error maps**
- These maps reveal:
  - Angular regions with strong observability
  - Regions with intrinsic degeneracy (e.g., mid-angle flattening)

---

### 2. Observation Angle Estimation

For each observation image:

1. Compute similarity against all templates
2. Estimate angle using three methods:
   - **Discrete estimate**: best-matching template
   - **Soft estimate**: similarity-weighted average of top-k matches
   - **Parabolic refinement**: local 3-point interpolation around the peak

Optional enhancements:
- ROI selection (mean-based or variance-based)
- Whitening / normalization
- Peak validity constraints for interpolation

---

## Results Summary

Key empirical findings:

- **Angular observability is clearly present** in raw side-view SPECT projections
- Even with **coarse angular templates**, sub-template precision is achievable
- Best performance is obtained with:
  - **5° template spacing**
  - Local parabolic refinement

Representative performance:

- Discrete estimate error: ~1–2°
- Soft / interpolated estimate error: **≈0.5°–1° (MAE)**

These results demonstrate that **continuous angular information is encoded in projection images**,
despite photon noise and limited sampling.

---

## Scope and Limitations

This demo intentionally focuses on observability rather than realism.

Not modeled:
- Attenuation and scatter
- Detector PSF
- Full collimator response
- Full 3D reconstruction

The goal is to establish a **quantitative upper bound on angular observability**.
More realistic physics and advanced inference methods are left for future work.

---

## Status

This repository represents a **completed feasibility milestone** demonstrating
that angle recovery from raw SPECT projections is both measurable and robust.

It serves as a foundation for:
- Adaptive or closed-loop acquisition strategies
- Lightweight angular self-calibration
- Integration with more realistic system models
