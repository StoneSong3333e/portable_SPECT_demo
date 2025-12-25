# Changelog
All notable changes to this project will be documented in this file.

[0.2.0] - 2025-12-6

Added
- Image-based feasibility demo for portable SPECT geometry estimation.
- Side-view SPECT raw projection simulation using official Jaszczak phantom data (Interfile .h33/.i33).
- Parallel-beam forward model with phantom rotation and line-integral projection.
- Poisson-count noise model to simulate realistic photon statistics.
- Feature-based angle estimation demo showing local angular observability from raw projections.
- Research roadmap documenting staged development from motion control to image-driven geometry inference.

Notes
- This release focuses on feasibility and observability analysis rather than full system calibration.
- Physics effects such as scatter, attenuation, and detector PSF are intentionally simplified.

## [0.1.1] - 2025-11-17
### Added
- Persistent JSON-based state storage for retaining rotary and linear positions across sessions.
- `save_state()` and `load_state()` helpers with automatic directory creation and safe default fallback.
- Atomic write helper (`_atomic_write`) using temporary-file replacement to prevent partial writes or corruption.
- Rotation normalization (`norm180`) integrated into the state-saving pipeline.

### Improved
- Increased robustness of the motion-control workflow through exception-safe state handling.
- Ensures consistent startup behavior by loading the previous position, reducing drift and improving reproducibility.

---

## [0.1.0] - 2025-10-29
### Added
- Initial public release of the motion-control + closed-loop framework.
- Rotary + linear stage control module.
- Basic sensor-feedback loop for angle and radius correction.
- Logging of drift and correction metrics.
