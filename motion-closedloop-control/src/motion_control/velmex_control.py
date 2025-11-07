
import time
import sys
import serial  # pip install pyserial
from detector_closed_loop import ClosedLoopController, FuncDetector, SimulatedDetector  #import self-detector


# ========== Basic control: VXM serial command wrapper ==========
class VXM:
    

    def __init__(self, port: str, baud: int = 9600, timeout: float = 0.5):     # command input as string

        # baud: serial baud rate (change according to your lab setup)
        # timeout: read timeout in seconds
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout) # open serial port

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def send(self, cmd: str, wait: float = 0.02) -> str:
        # Commands must end with '\r'
        if not cmd.endswith('\r'):
            cmd += '\r'
        # Common pitfall: missing carriage return will prevent command execution
        self.ser.write(cmd.encode('ascii'))
        # Core: encode cmd to ASCII bytes and send via .write() to the serial port (e.g., COM7).

        time.sleep(wait)
        # give time for echo
        resp = self.ser.read(self.ser.in_waiting or 1)  # read all available bytes or at least one byte to avoid blocking
        return resp.decode(errors='ignore')  # ignore undecodable characters

    def ready(self) -> bool:
        # Is the controller idle? (R = ready, B = busy)
        r = self.send('V')
        return 'R' in r

    def wait_until_ready(self, poll: float = 0.05, timeout: float = 120.0):
        # Poll readiness every `poll` seconds until timeout
        t0 = time.time()
        while True:
            if self.ready():
                return
            if time.time() - t0 > timeout:
                raise TimeoutError("Timeout waiting for controller to become ready")
            time.sleep(poll)

    # ---- Common atomic actions ----
    def clear_program(self):
        self.send('C')  # Clear buffer

    def run(self):
        self.send('R')  # Run
        # execute buffered commands

    def stop(self):
        self.send('D')  # Decelerate and stop

    def kill(self):
        self.send('K')  # Emergency stop

    def set_speed_accel(self, m: int, speed: int = 800, accel: int = 30, full_power: bool = True):
        # Speed unit = half-steps/sec; accel: 1~127; for first connection use low speed (600~1200)
        # VXM treats parameter-setting as buffered commands; parameter settings take effect immediately without needing R
        self.send(f'{"SA" if full_power else "S"}{m}M{speed}')
        self.send(f'A{m}M{accel}')

    def move_abs_steps(self, m: int, steps: int):
        # Absolute positioning (relative to software zero): IA m M steps
        self.clear_program()
        self.send(f'IA{m}M{steps}')
        self.run()
        self.wait_until_ready()

    def move_inc_steps(self, m: int, dsteps: int):
        # Incremental move: I m M dsteps (can be positive or negative)
        self.clear_program()
        self.send(f'I{m}M{dsteps}')
        self.run()
        self.wait_until_ready()

    def home_to_negative_limit(self, m: int):
        # Find reference (negative limit) -> set current position as absolute 0:
        #   I m M -0 : search negative limit
        #   IA m M -0: set current position to absolute 0
        # Search negative limit
        self.clear_program()
        self.send(f'I{m}M-0')
        self.run()
        self.wait_until_ready()

    # set current position as absolute 0
        self.clear_program()
        self.send(f'IA{m}M-0')
        self.run()
        self.wait_until_ready()

    def query_position_steps(self, m: int = 1) -> int:
        # Query current position in steps (if firmware echoes). Axis 1=X, 2=Y, 3=Z, 4=T
        # Echo format may differ; if parsing fails return 0 which does not affect basic usage.
        
        letter = {1: 'X', 2: 'Y', 3: 'Z', 4: 'T'}[m]
        resp = self.send(letter)
        digits = ''.join(ch for ch in resp if (ch.isdigit() or ch == '-'))
        try:
            return int(digits)
        except ValueError:
            return 0

# ========== Rotary stage: B4872TS (72:1) angle wrapper ==========
class B48Stage72:
    """
    Rotary stage logic layer:
    - All commands ultimately become VXM move_inc_steps(...) / move_abs_steps(...)
    - Maintains two angle trackers:
        1) accum_deg: accumulated angle (not normalized), used to detect excessive movement in one direction
        2) current_deg: normalized current angle (mapped to (-180,180]), for display
    - Core rule: if accumulated movement in the same direction exceeds dir_limit_deg after a move,
      perform a corrective reverse 360° turn before executing the requested move.
    - New behavior:
      If enable_homing = True -> attempt mechanical homing on startup.
      If homing fails (no limit switch / timeout) -> disable homing automatically; user should manually
      align the stage and call set_zero().
    """

    def __init__(
        self,
        vxm: VXM,
        motor: int,
        soft_min_deg: float = -180.0,
        soft_max_deg: float = 180.0,
        dir_limit_deg: float = 181.0,
        steps_per_rev: int = 200 * 16 * 72,   # needs lab confirmation for microstepping
        enable_homing: bool = True,           # enable homing on startup for this rotary channel
    ):
        self.vxm = vxm
        self.m = motor
        self.soft_min = soft_min_deg
        self.soft_max = soft_max_deg
        self.dir_limit = dir_limit_deg
        self.steps_per_rev = steps_per_rev
        self.enable_homing = bool(enable_homing)

        # Accumulated angle, not normalized, used to check same-direction accumulation
        self.accum_deg = 0.0
        # Current angle, normalized for display
        self.current_deg = 0.0

    # ----- Utility functions -----
    def _normalize(self, deg: float) -> float:
        # Map any angle into the range (-180, 180]
        return ((deg + 180) % 360) - 180

    def _deg_to_steps(self, deg: float) -> int:
        # Degrees -> motor steps
        steps = deg / 360.0 * self.steps_per_rev
        return int(round(steps))

    # ----- Startup: home if enabled -----
    def home_if_enabled(self):
        """
        Recommended to call at startup:
        - enable_homing=True -> perform mechanical homing
        - if homing fails or times out -> disable homing automatically and prompt user to set software zero
        - enable_homing=False -> skip homing
        """
        if not self.enable_homing:
            print("[ROT] homing disabled → please manually move to reference position then call set_zero()")
            return

        ok = self.home(timeout=5.0)
        if not ok:
            self.enable_homing = False
            print("[ROT] mechanical homing failed for this channel -> homing disabled, please manually align and call set_zero()")

    # ----- Mechanical homing -----
    def home(self, timeout: float = 5.0) -> bool:
        """
        Perform mechanical homing using VXM negative-limit search:
        1) I m M -0 : search negative limit
        2) IA m M -0: set current position to absolute 0
        If the controller remains busy until timeout, assume no limit switch and return False.
        """
    # 1) search for negative limit
        self.vxm.clear_program()
        self.vxm.send(f'I{self.m}M-0')
        self.vxm.run()

        t0 = time.time()
        while True:
            if self.vxm.ready():
                # 2) set current position as absolute 0
                self.vxm.clear_program()
                self.vxm.send(f'IA{self.m}M-0')
                self.vxm.run()
                self.vxm.wait_until_ready()
                # reset software angles as well
                self.accum_deg = 0.0
                self.current_deg = 0.0
                print("[ROT] homing done → angle = 0.0° (mechanical)")
                return True

            if time.time() - t0 > timeout:
                # Timeout: assume no limit switch
                self.accum_deg = 0.0
                self.current_deg = 0.0
                print("[ROT] homing timeout → likely no limit switch attached, please manually move to reference and call set_zero()")
                return False

            time.sleep(0.05)

    # ----- Software zero (for manual alignment) -----
    def set_zero(self):
        """
        After manually rotating the stage to the desired software zero, call this.
        Does not move the motor; only resets the software angle to zero.
        """
        self.accum_deg = 0.0
        self.current_deg = 0.0
        print("[ROT] software zero → 0.0°")

    # ----- Core: movement with direction protection -----
    def _move_with_dir_limit(self, delta_deg: float):
        """
        Core movement routine used by both absolute and relative moves.
        Logic:
        1) Predict if accumulated same-direction movement will exceed ±dir_limit after this step.
        2) If it would exceed, perform a corrective reverse 360° before executing the requested move.
        3) Update accum_deg and current_deg after the move.
        """
    # 1) predict the accumulated angle after this move
        predicted = self.accum_deg + delta_deg

        # 2) Check whether this would exceed the same-direction limit
        if abs(predicted) > self.dir_limit:
            # decide which direction to correct by sign of predicted
            correction = -360.0 if predicted > 0 else 360.0
            steps_corr = self._deg_to_steps(correction)
            # first reverse one full turn (360°)
            self.vxm.move_inc_steps(self.m, steps_corr)
            # also pull the accumulated angle back
            self.accum_deg += correction
            self.current_deg = self._normalize(self.accum_deg)
            print(f"[ROT] PROTECT: same-direction accumulation would exceed ±{self.dir_limit}° -> first reverse {correction:+.1f}° -> accum={self.accum_deg:+.2f}°")

    # 3) Execute the actual requested move
        steps_move = self._deg_to_steps(delta_deg)
        self.vxm.move_inc_steps(self.m, steps_move)

    # 4) update the software-tracked angles
        self.accum_deg += delta_deg
        self.current_deg = self._normalize(self.accum_deg)


    # ----- Public: absolute angle -----
    def goto_deg(self, target_deg: float):
        # Normalize to (-180,180]
        target_norm = self._normalize(target_deg)
    # amount to move
        delta = target_norm - self.current_deg
        self._move_with_dir_limit(delta)

    # ----- Public: relative angle -----
    def jog_deg(self, delta_deg: float):
        self._move_with_dir_limit(delta_deg)
    
        # --- Getter for detector/truth (deg) ---
    @property
    def angle_deg(self) -> float:
        """Current software-tracked angle in degrees."""
        return float(self.current_deg)

# ========== Linear axis: Velmex MA60 + PK266-03A-P1 ==========
class LinearAxis:
    """
    Wrapper for Velmex MA60 linear module control.
    - Motor: PK266-03A-P1 (1.8°/step -> 200 steps/rev)
    - lead_mm_per_rev: screw lead in mm per revolution (confirm in lab)
    - microstep: VXM microstepping setting (confirm in lab)
    - New: enable_homing option: attempt mechanical homing if available, otherwise use software zero
    """

    def __init__(self, ctl: VXM, motor: int,
                 lead_mm_per_rev: float,      # screw lead in mm/rev
                 microstep: int = 16,         # microstep setting; check VXM DIP switches
                 soft_min_mm: float = 0.0,    # soft travel minimum (mm)
                 soft_max_mm: float = 150.0,  # soft travel maximum (mm)
                 enable_homing: bool = True   # enable mechanical homing on startup
                 ):
        self.ctl = ctl
        self.m = motor
        self.lead = float(lead_mm_per_rev)
        self.microstep = int(microstep)

        motor_steps_per_rev = 200               # 1.8° -> 200 steps/rev
        self.steps_per_rev = motor_steps_per_rev * self.microstep
        self.steps_per_mm = self.steps_per_rev / self.lead

        self.soft_min = float(soft_min_mm)
        self.soft_max = float(soft_max_mm)
        self.current_mm = 0.0                   # software current position (mm)
        self.enable_homing = bool(enable_homing)

    # ---- Utility: mm -> steps ----
    def _mm_to_steps(self, mm: float) -> int:
        return int(round(mm * self.steps_per_mm))

    # ---- Startup: home_if_enabled ----
    def home_if_enabled(self):
        if not self.enable_homing:
            print("[LIN] homing disabled → please manually move to reference position then call set_zero()")
            return

        ok = self.home(timeout=5.0)
        if not ok:
            self.enable_homing = False
            print("[LIN] mechanical homing not supported for this channel -> homing disabled, please manually move to limit and call set_zero()")

    # ---- Mechanical homing ----
    def home(self, timeout: float = 5.0) -> bool:
        """
        Use VXM negative-limit search to reach mechanical reference, then clear software position to 0.
        If timeout occurs assume no limit switch and return False.
        """
    # send negative-limit search
        self.ctl.clear_program()
        self.ctl.send(f'I{self.m}M-0')
        self.ctl.run()

        t0 = time.time()
        while True:
            if self.ctl.ready():
                # after successful homing, set current position as absolute 0
                self.ctl.clear_program()
                self.ctl.send(f'IA{self.m}M-0')
                self.ctl.run()
                self.ctl.wait_until_ready()
                self.current_mm = 0.0
                print("[LIN] homing done → pos = 0.0 mm (mechanical)")
                return True

            if time.time() - t0 > timeout:
                # Timeout -> likely no limit switch attached
                self.current_mm = 0.0
                print("[LIN] homing timeout → likely no limit switch; please manually move to physical reference and call set_zero()")
                return False

            time.sleep(0.05)

    # ---- Software zero ----
    def set_zero(self):
        """Set software zero only; does not move hardware."""
        self.current_mm = 0.0
        print("[LIN] software zero set to 0.0 mm")

    # ---- Absolute positioning ----
    def goto_mm(self, target_mm: float):
        """
        Move to absolute position target_mm (relative to software zero)
        """
        if not (self.soft_min <= target_mm <= self.soft_max):
            raise ValueError(f"Target {target_mm:.3f} mm outside soft limits [{self.soft_min},{self.soft_max}] mm")
        steps = self._mm_to_steps(target_mm)
        self.ctl.move_abs_steps(self.m, steps)
        self.current_mm = target_mm
        print(f"[LIN] goto {target_mm:.3f} mm → {steps} steps")

    # ---- Relative positioning ----
    def jog_mm(self, delta_mm: float):
        """
        Relative move: positive -> forward, negative -> backward
        """
        target_mm = self.current_mm + delta_mm
        if not (self.soft_min <= target_mm <= self.soft_max):
            raise ValueError(f"Target after increment {target_mm:.3f} mm outside soft limits")
        dsteps = self._mm_to_steps(delta_mm)
        self.ctl.move_inc_steps(self.m, dsteps)
        self.current_mm = target_mm
        print(f"[LIN] jog {delta_mm:+.3f} mm → {dsteps:+d} steps → now {self.current_mm:.3f} mm")


        # --- Getter for detector/truth (mm) ---
    @property
    def pos_mm(self) -> float:
        """Current software-tracked position in mm."""
        return float(self.current_mm)


# ========== High-level path planning / point scheduler ==========
class PositionPlanner:
    """
    Higher-level controller:
    - Manages two lower-level axes: rot (rotary) and lin (linear)
    - Maps (soft zero index, angle index, radius index) to actual (deg, mm)
    - Supports 4 * 12 * 5 = 240 positions
    """

    def __init__(
        self,
        rot: B48Stage72,
        lin: LinearAxis,
        soft_zeros_deg=None,
        angle_step_deg: float = 30.0,
        num_angles: int = 12,
        radii_mm=None,
        cl=None,
    ):
        self.rot = rot
        self.lin = lin
        self.cl = cl

        # Four different software zeros, default 0 / 90 / 180 / 270
        if soft_zeros_deg is None:
            soft_zeros_deg = [0.0, 90.0, 180.0, 270.0]
        self.soft_zeros = soft_zeros_deg

    # How many angular steps per full circle (default 30° * 12 = 360°)
        self.angle_step = float(angle_step_deg)
        self.num_angles = int(num_angles)

        # Five radii in mm; placeholder values — replace with actual lab values
        if radii_mm is None:
            radii_mm = [0.0, 20.0, 40.0, 60.0, 80.0]
        self.radii = radii_mm

    # ---- Utility: check indices ----
    def _check_indices(self, soft_idx: int, angle_idx: int, radius_idx: int):
        if not (0 <= soft_idx < len(self.soft_zeros)):
            raise ValueError(f"soft zero index {soft_idx} out of range 0~{len(self.soft_zeros)-1}")
        if not (0 <= angle_idx < self.num_angles):
            raise ValueError(f"angle index {angle_idx} out of range 0~{self.num_angles-1}")
        if not (0 <= radius_idx < len(self.radii)):
            raise ValueError(f"radius index {radius_idx} out of range 0~{len(self.radii)-1}")
    













    # ---- Core: move to a point ----
    def goto_point(self, soft_idx: int, angle_idx: int, radius_idx: int):
        """
        Parameters:
        - soft_idx  : 0~3  one of the four software zeros
        - angle_idx : 0~11 angular index (step = angle_step)
        - radius_idx: 0~4  radius index
        """
        self._check_indices(soft_idx, angle_idx, radius_idx)

        base_deg = self.soft_zeros[soft_idx]
        add_deg = angle_idx * self.angle_step
        target_deg = base_deg + add_deg

        target_mm = self.radii[radius_idx]

        # You can choose to move rotary or linear first; here we rotate first
        print(f"[PLAN] goto S{soft_idx} / A{angle_idx} / R{radius_idx} → rot={target_deg:.2f}°, lin={target_mm:.2f} mm")
        if self.cl is not None:
            self.cl.move_and_trim(target_deg, target_mm)
        else:
            self.rot.goto_deg(target_deg)
            self.lin.goto_mm(target_mm)
        
    # ---- Scan all 4*12*5 ----
    def scan_all(self):
        """
        Iterate through all positions. Order: soft-zero -> angle -> radius
        """
        for s in range(len(self.soft_zeros)):
            for a in range(self.num_angles):
                for r in range(len(self.radii)):
                    self.goto_point(s, a, r)


# ---- Adapters: map your goto/jog API to the closed-loop module's minimal API ----
class RotAdapter:
    """Expose move_to_deg / move_by_deg for the rotary stage."""
    def __init__(self, rot: B48Stage72):
        self._rot = rot
    def move_to_deg(self, deg: float):
        # absolute command -> use your goto_deg
        return self._rot.goto_deg(deg)
    def move_by_deg(self, ddeg: float):
        # relative command -> use your jog_deg
        return self._rot.jog_deg(ddeg)
    @property
    def angle_deg(self) -> float:
        return self._rot.angle_deg

class LinAdapter:
    """Expose move_to_mm / move_by_mm for the linear stage."""
    def __init__(self, lin: LinearAxis):
        self._lin = lin
    def move_to_mm(self, mm: float):
        return self._lin.goto_mm(mm)
    def move_by_mm(self, dmm: float):
        return self._lin.jog_mm(dmm)
    @property
    def pos_mm(self) -> float:
        return self._lin.pos_mm






# ========== Demo main program ==========
if __name__ == "__main__":
    COM_PORT = "COM5"   # <- change this to your serial port
    BAUD = 9600
    ROT_MOTOR = 1
    LIN_MOTOR = 2

    vxm = VXM(port=COM_PORT, baud=BAUD, timeout=0.5)

    try:
        # 1) Set speed and acceleration (optional)
        vxm.set_speed_accel(m=ROT_MOTOR, speed=800, accel=30, full_power=True)
        vxm.set_speed_accel(m=LIN_MOTOR, speed=800, accel=30, full_power=True)

        # 2) Initialize two axes
        rot = B48Stage72(
            vxm,
            motor=ROT_MOTOR,
            soft_min_deg=-180,
            soft_max_deg=180,
            dir_limit_deg=181,
            steps_per_rev=200 * 16 * 72,   # <- confirm microstepping on site
            enable_homing=True
        )

        linear = LinearAxis(
            vxm,
            motor=LIN_MOTOR,
            lead_mm_per_rev=2.54,      # <- screw lead (mm/rev), confirm in lab
            microstep=16,              # <- confirm microstep setting
            soft_min_mm=0.0,
            soft_max_mm=150.0,
            enable_homing=True
        )

        # 3) Homing procedure (home if available, otherwise use software zero)
        print("[INFO] Homing rotary stage...")
        rot.home_if_enabled()

        print("[INFO] Homing linear stage...")
        linear.home_if_enabled()
                # 4.5) Closed-loop wiring (simulated detector for now)
        # Wrap axes to match the closed-loop controller's minimal API


        rot_ctl = RotAdapter(rot)
        lin_ctl = LinAdapter(linear)

        # Simulated detector truth: read current software-tracked pose
        def _truth():
            # angle in deg, radius in mm
            return (rot.angle_deg, linear.pos_mm)

        # Use simulated detector with small bias/noise/dropouts
        det = SimulatedDetector(
            ground_truth_fn=_truth,
            theta_bias_deg=0.15,
            radius_bias_mm=-0.03,
            theta_noise_std_deg=0.06,
            radius_noise_std_mm=0.02,
            dropout_prob=0.05
        )

        # Build the closed-loop controller (tune to taste)
        cl = ClosedLoopController(
            rot_ctl, lin_ctl, det,
            tol_theta_deg=0.20,
            tol_radius_mm=0.05,
            kp_theta=1.0,
            kp_radius=1.0,
            max_step_theta_deg=2.0,
            max_step_radius_mm=0.5,
            max_corrections=3,
            settle_wait_s=0.10,
            verbose=True
        )

        # 4) High-level planner
        planner = PositionPlanner(
            rot,
            linear,
            soft_zeros_deg=[0.0, 90.0, 180.0, 270.0],   # <- four software zeros, default 90° apart
            angle_step_deg=30.0,                        # <- 30° per angular step
            num_angles=12,                              # <- 12 points per full circle
            radii_mm=[0.0, 25.0, 50.0, 75.0, 100.0],    # <- five radii, replace with lab-verified values
            cl=cl                                       # <-- NEW: give planner the closed-loop controller
        )

        # 5) Example: go to "soft zero #2 (90°) / angle #4 (3*30=90°) / radius #3 (50 mm)"
        # This results in rotation = 90 + 90 = 180°, linear = 50 mm
        print("[INFO] Move to one planned point ...")
        planner.goto_point(soft_idx=1, angle_idx=3, radius_idx=2)

        # 6) (optional) scan all 4*12*5 = 240 points
        # !!! WARNING: this will take a long time; only enable if you have confirmed sufficient safe workspace
        # print("[INFO] Scan all 240 points ...")
        # planner.scan_all()

        print("[INFO] Motion complete.")

    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            vxm.kill()
        except Exception:
            pass
        raise
    finally:
        vxm.close()
        print("[INFO] Serial closed.")
