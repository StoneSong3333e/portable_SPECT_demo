# Portable SPECT: Image-Driven Geometry & Adaptive Acquisition

This repository hosts an ongoing research and engineering project
exploring **image-driven geometry inference** and **adaptive acquisition**
for compact and portable SPECT systems.

The long-term goal is to reduce reliance on high-precision mechanical hardware
by leveraging **information already present in raw SPECT projection data**.

---

## Motivation

Conventional SPECT systems assume that detector geometry
(angles, positions, trajectories)
must be known with high mechanical precision.

This project explores an alternative question:

> Can geometric information be inferred directly from
> photon-counting projection data itself?

If feasible, this would enable:
- Lighter and more portable SPECT designs
- Reduced calibration burden
- Image-driven self-correction and adaptive scanning strategies

---

## What this repository contains

This repository combines **hardware control**, **physics simulation**,
and **image-based analysis** under a single research framework.

At a high level, it includes:

- Motion-control and closed-loop correction infrastructure
- Physics-based simulation tools for SPECT projection generation
- Image-based feasibility studies on geometric observability
- Lightweight analysis pipelines for quantitative evaluation

The codebase is structured to separate
**baseline engineering infrastructure**
from **image- and physics-driven research experiments**.

Not all directories are required for every demo;
each component is designed to be usable independently.

---

## Current Research Focus

The current primary focus is **angular observability from raw SPECT projections**.

Key questions being studied:
- Does side-view SPECT projection data encode recoverable angle information?
- How does angular observability degrade with coarser angular sampling?
- What precision is achievable under realistic photon-counting noise?

These questions are investigated using:
- Physics-based Geant4 simulations
- Custom-designed simplified activity phantoms
- Image-only matching and interpolation (no reconstruction)

---

## Key Findings (So Far)

- Raw SPECT projections contain **measurable and continuous angular information**
- Angle can be estimated to **sub-template precision**
  even under realistic Poisson noise
- With moderate template density (e.g., 5° spacing),
  refined estimates achieve **~0.5°–1° mean absolute error**

These results establish a quantitative **observability baseline**
for future adaptive or closed-loop SPECT systems.

---

## Scope and Philosophy

This repository emphasizes:
- Feasibility
- Observability
- Reproducibility

It intentionally avoids:
- Full clinical realism
- Complete tomographic reconstruction
- Hardware-specific optimization

The goal is to answer **whether something is possible**
before optimizing **how well it can be done**.

---

## Status

This project is under active development.

Several feasibility milestones have been completed,
with results documented through reproducible simulations
and lightweight analysis pipelines.

Future work will expand toward:
- More realistic physics models
- Adaptive acquisition strategies
- Integration with hardware-in-the-loop experiments
