# Research Roadmap

This document outlines the staged research direction of this repository.
Only completed and actively developed components are implemented in code;
later phases represent planned or exploratory research directions.

---

## Phase 0 – Sensor-Based Closed-Loop Motion Control (Completed)

Baseline motion control for rotary and linear stages using external sensors.
This phase provides a hardware-precision reference system, including
error logging, bounded correction, and state persistence.

---

## Phase I – Angle Observability from Raw SPECT Projections (Completed)

Feasibility study demonstrating that side-view SPECT raw projections
retain recoverable angular information under Poisson counting noise.

Key elements:
- Official Jaszczak phantom activity distributions
- Side-view projection via 3D rotation and line integrals
- Small angular perturbations simulating mechanical uncertainty
- Feature-based matching to estimate local angle deviation

This phase establishes an upper-bound feasibility result rather than
a finalized calibration solution.

---

## Phase II – Physics-Realistic Validation with Geant4 (Planned)

Repeat the observability analysis using Geant4/GATE simulations with Tc-99m
and conventional scintillation detectors.

To control complexity, initial studies will focus on 2D Jaszczak phantoms
while incorporating more realistic physical effects.

---

## Phase III – Low-Count and 3D Phantom Studies (Future Work)

Extend the analysis to full 3D Jaszczak phantoms, including scatter,
energy windowing, and low-count acquisition regimes.

The goal is to quantify the limits of geometric observability under
near-clinical conditions.

---

## Phase IV – Improved Image-Based Geometry Estimation (Exploratory)

Develop more robust image-analysis strategies to improve angular and
radial estimation accuracy, particularly under limited photon statistics.

---

## Phase V – Geometry-Aware Reconstruction (Exploratory)

Investigate reconstruction workflows in which image-derived geometric
estimates are incorporated into the forward model of iterative SPECT
reconstruction, reducing reliance on high-precision mechanical encoders.

---

## Phase VI – Adaptive Scan Planning (Optional Extension)

Explore adaptive acquisition strategies, such as Frank–Wolfe–based
optimization, to select projection angles and positions that maximize
information gain under time and dose constraints.

---

## Long-Term Vision

An image-adaptive, computationally enhanced portable SPECT system that
reduces hardware precision requirements while maintaining clinically
useful image quality.
