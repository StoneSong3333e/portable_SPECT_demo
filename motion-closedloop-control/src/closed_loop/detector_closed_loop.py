"""
Closed-loop trim module (angle + radius) with a pluggable detector.

Usage patterns:
1) As a library in your project
   ----------------------------------------------------------------
   from detector_closed_loop import ClosedLoopController, FuncDetector

   # rot/lin are your stage objects; they only need the minimal API:
   #   rot.move_to_deg(x), rot.move_by_deg(dx)
   #   lin.move_to_mm(x),  lin.move_by_mm(dx)
   det = FuncDetector(fetch_pose_fn=my_fetch_pose)  # <-- replace with your detector hook
   cl  = ClosedLoopController(rot, lin, det, tol_theta_deg=0.2, tol_radius_mm=0.05)

   cl.move_and_trim(theta_cmd_deg=150.0, radius_cmd_mm=50.0)

2) Run this file directly for a self-contained demo
   ----------------------------------------------------------------
   python detector_closed_loop.py
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import math
import random
import time


# -------------------------
# Detector interfaces
# -------------------------

class DetectorBase:
    """
    Abstract detector interface.
    Implement read_pose() so that it returns the measured (theta_deg, radius_mm),
    or None if the detector cannot provide a reading for this cycle.

    Replace the SimulatedDetector with your real implementation, or use FuncDetector
    to inject a simple callback (fetch_pose_fn) that returns a tuple.
    """
    def read_pose(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError


class FuncDetector(DetectorBase):
    """
    Thin adapter that turns a function into a detector.

    fetch_pose_fn must return either (theta_deg, radius_mm) or None.
    """
    def __init__(self, fetch_pose_fn: Callable[[], Optional[Tuple[float, float]]]):
        self._fn = fetch_pose_fn

    def read_pose(self) -> Optional[Tuple[float, float]]:
        return self._fn()


class SimulatedDetector(DetectorBase):
    """
    A simple simulated detector.
    - It queries ground truth from a callback (ground_truth_fn), which should
      return the current mechanical (theta_deg, radius_mm).
    - It adds bias and gaussian noise to emulate measurement imperfection.
    - You can also emulate occasional dropouts (returning None).

    Replace this with your real detector later.
    """
    def __init__(
        self,
        ground_truth_fn: Callable[[], Tuple[float, float]],
        theta_bias_deg: float = 0.15,
        radius_bias_mm: float = -0.03,
        theta_noise_std_deg: float = 0.06,
        radius_noise_std_mm: float = 0.02,
        dropout_prob: float = 0.05
    ):
        self._truth = ground_truth_fn
        self.theta_bias_deg = theta_bias_deg
        self.radius_bias_mm = radius_bias_mm
        self.theta_noise_std_deg = theta_noise_std_deg
        self.radius_noise_std_mm = radius_noise_std_mm
        self.dropout_prob = dropout_prob

    def read_pose(self) -> Optional[Tuple[float, float]]:
        # emulate occasional missing measurement
        if random.random() < self.dropout_prob:
            return None

        theta, radius = self._truth()
        theta_meas = theta + self.theta_bias_deg + random.gauss(0.0, self.theta_noise_std_deg)
        radius_meas = radius + self.radius_bias_mm + random.gauss(0.0, self.radius_noise_std_mm)
        # wrap angle to [0, 360)
        theta_meas = theta_meas % 360.0
        return (theta_meas, radius_meas)


# -------------------------
# Closed-loop controller
# -------------------------

class ClosedLoopController:
    """
    Wraps a rotary (angle) axis and a linear (radius) axis.
    After moving to (theta_cmd, radius_cmd), it queries the detector,
    compares the measured pose, and applies bounded proportional trims.

    Minimal axis API expected:
      rot.move_to_deg(x), rot.move_by_deg(dx)
      lin.move_to_mm(x),  lin.move_by_mm(dx)
    """
    def __init__(
        self,
        rot, lin, detector: DetectorBase,
        *,
        tol_theta_deg: float = 0.20,
        tol_radius_mm: float = 0.05,
        kp_theta: float = 1.0,
        kp_radius: float = 1.0,
        max_step_theta_deg: float = 2.0,
        max_step_radius_mm: float = 0.5,
        max_corrections: int = 3,
        settle_wait_s: float = 0.10,
        verbose: bool = True
    ):
        self.rot = rot
        self.lin = lin
        self.detector = detector

        self.tol_theta_deg = tol_theta_deg
        self.tol_radius_mm = tol_radius_mm
        self.kp_theta = kp_theta
        self.kp_radius = kp_radius
        self.max_step_theta_deg = max_step_theta_deg
        self.max_step_radius_mm = max_step_radius_mm
        self.max_corrections = max_corrections
        self.settle_wait_s = settle_wait_s
        self.verbose = verbose

    @staticmethod
    def _wrap_err_deg(err: float) -> float:
        """
        Map an angle error to (-180, 180].
        This avoids 359 -> 0 being treated as -359 error, etc.
        """
        return (err + 180.0) % 360.0 - 180.0

    def move_and_trim(self, theta_cmd_deg: float, radius_cmd_mm: float) -> dict:
        """
        1) Move to commanded pose (open-loop).
        2) Query detector and trim both axes until within tolerance or attempts exhausted.

        Returns a summary dict for logging/diagnostics.
        """
        # Open-loop move first
        self.rot.move_to_deg(theta_cmd_deg)
        self.lin.move_to_mm(radius_cmd_mm)

        # Allow the mechanism to settle (belts, backlash, etc.)
        time.sleep(self.settle_wait_s)

        attempts = 0
        last_meas = None
        last_err = None

        while attempts < self.max_corrections:
            pose = self.detector.read_pose()
            if pose is None:
                # No data this round; wait and retry
                attempts += 1
                if self.verbose:
                    print(f"[trim] detector dropout; retry {attempts}/{self.max_corrections}")
                time.sleep(self.settle_wait_s)
                continue

            theta_meas, radius_meas = pose
            e_theta = self._wrap_err_deg(theta_cmd_deg - theta_meas)
            e_radius = radius_cmd_mm - radius_meas
            last_meas = (theta_meas, radius_meas)
            last_err = (e_theta, e_radius)

            within_theta = abs(e_theta) <= self.tol_theta_deg
            within_radius = abs(e_radius) <= self.tol_radius_mm

            if self.verbose:
                print(
                    f"[trim] meas θ={theta_meas:.3f}°, r={radius_meas:.3f} mm | "
                    f"err dθ={e_theta:+.3f}°, dr={e_radius:+.3f} mm"
                )

            if within_theta and within_radius:
                if self.verbose:
                    print("[trim] within tolerance; done.")
                break

            # Proportional step with bounding (small bounded nudge is safer than a jump)
            d_theta = max(-self.max_step_theta_deg, min(self.kp_theta * e_theta, self.max_step_theta_deg))
            d_radius = max(-self.max_step_radius_mm, min(self.kp_radius * e_radius, self.max_step_radius_mm))

            # Only move the axis that is out of tolerance
            if not within_theta:
                self.rot.move_by_deg(d_theta)
                if self.verbose:
                    print(f"[trim] apply Δθ={d_theta:+.3f}°")
            if not within_radius:
                self.lin.move_by_mm(d_radius)
                if self.verbose:
                    print(f"[trim] apply Δr={d_radius:+.3f} mm")

            attempts += 1
            time.sleep(self.settle_wait_s)

        return {
            "cmd": (theta_cmd_deg, radius_cmd_mm),
            "meas": last_meas,
            "err": last_err,
            "attempts": attempts
        }


# -------------------------
# Minimal mock axes (for demo)
# -------------------------

class _MockRot:
    """
    Minimal mock of a rotary stage.
    - move_to_deg / move_by_deg update an internal angle state with wrap.
    - a tiny first-order lag is emulated so the "ground truth" changes smoothly.
    """
    def __init__(self, init_deg: float = 0.0):
        self._angle_deg = init_deg % 360.0

    @property
    def angle_deg(self) -> float:
        return self._angle_deg % 360.0

    def move_to_deg(self, deg: float) -> None:
        target = deg % 360.0
        # emulate that the motor reaches the target quickly
        self._angle_deg = target

    def move_by_deg(self, ddeg: float) -> None:
        self._angle_deg = (self._angle_deg + ddeg) % 360.0


class _MockLin:
    """
    Minimal mock of a linear stage.
    """
    def __init__(self, init_mm: float = 0.0):
        self._pos_mm = float(init_mm)

    @property
    def pos_mm(self) -> float:
        return self._pos_mm

    def move_to_mm(self, mm: float) -> None:
        self._pos_mm = float(mm)

    def move_by_mm(self, dmm: float) -> None:
        self._pos_mm += float(dmm)


# -------------------------
# Self-test / demo
# -------------------------

def _demo():
    print("=== Closed-loop demo with simulated detector ===")
    rot = _MockRot(init_deg=10.0)
    lin = _MockLin(init_mm=20.0)

    # Ground truth is read from the mock axes:
    def truth():
        return (rot.angle_deg, lin.pos_mm)

    # Simulated detector with small bias/noise
    det = SimulatedDetector(
        ground_truth_fn=truth,
        theta_bias_deg=0.18,
        radius_bias_mm=-0.04,
        theta_noise_std_deg=0.05,
        radius_noise_std_mm=0.02,
        dropout_prob=0.10
    )

    cl = ClosedLoopController(
        rot, lin, det,
        tol_theta_deg=0.20,     # angle tolerance
        tol_radius_mm=0.05,     # radius tolerance
        kp_theta=0.9,           # proportional gain for angle
        kp_radius=0.9,          # proportional gain for radius
        max_step_theta_deg=1.5, # max nudge per trim
        max_step_radius_mm=0.30,
        max_corrections=4,
        settle_wait_s=0.10,
        verbose=True
    )

    # Try a few targets
    targets = [
        (150.0, 50.0),
        (352.0, 30.0),   # exercise angle wrap near 360
        (5.0,   75.0),
    ]

    for theta_cmd, radius_cmd in targets:
        print(f"\n--> Go to θ={theta_cmd}°, r={radius_cmd} mm")
        res = cl.move_and_trim(theta_cmd, radius_cmd)
        print(f"    summary: {res}")

    print("\nDemo done.\n")


if __name__ == "__main__":
    _demo()