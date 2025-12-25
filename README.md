# Portable SPECT: Image-Driven Geometry Feasibility Studies

This repository collects research and engineering efforts toward
image-driven geometry inference and adaptive acquisition
for compact and portable SPECT systems.

The goal is to explore whether raw SPECT projection data
contains recoverable geometric information that can be used
to reduce dependence on high-precision mechanical hardware.

---

## What this repository contains

This repository currently includes:

- Sensor-based motion control and closed-loop correction modules
  (rotary and linear stages, error logging, and recovery mechanisms)
- Image-based feasibility demos testing geometric observability
  from SPECT raw projection data
- Documentation of planned research directions and system extensions

The code is organized to separate baseline control infrastructure
from physics- and image-driven feasibility studies.

---

## Current focus

The current technical focus is on **angle observability from raw SPECT projections**:

- Using official Jaszczak phantom activity distributions
- Simulating side-view SPECT projections via 3D rotation and line integrals
- Injecting small angular perturbations and realistic Poisson counting noise
- Evaluating whether projection data retains enough information
  to estimate local angular deviations through image analysis

This work serves as an early-stage feasibility study rather than
a finalized calibration or reconstruction algorithm.

---

## Demos

- **SPECT Angle Observability (Jaszczak Phantom)**  
  A reproducible demo showing that side-view SPECT raw projections
  retain recoverable angle-dependent information under Poisson noise.  
  See `demos/spect_angle_observability/`.

---

## Status

This repository reflects an ongoing research effort.
Completed components, active experiments, and planned extensions
are documented incrementally as the project evolves.
