import numpy as np
from pathlib import Path
from scipy.ndimage import rotate
import imageio.v2 as imageio

# ---------- 1. Load Jaszczak phantom ----------
def load_h33_header(header_path):
    header = {}
    with open(header_path, "r") as f:
        for line in f:
            if ":=" not in line:
                continue
            key, val = line.split(":=", 1)
            key = key.strip()
            val = val.strip()
            if key.startswith("!"):
                key = key[1:].strip()
            header[key] = val
    return header

def load_i33_volume(header_path):
    header = load_h33_header(header_path)
    nx = int(header["matrix size [1]"])
    ny = int(header["matrix size [2]"])
    nz = int(header["number of slices"])
    data_file = header["name of data file"]
    data_path = Path(header_path).with_name(data_file)

    arr = np.fromfile(data_path, dtype=np.uint16)
    expected = nx * ny * nz
    if arr.size != expected:
        raise RuntimeError(f"Data length mismatch: got {arr.size}, expected {expected}")
    vol = arr.reshape((nz, ny, nx))   # (Z, Y, X)
    return vol

# ---------- 2. Side projection: camera at +X, phantom rotates around Z ----------

def project_side(volume, angle_deg):
    """
    volume: 3D phantom, shape (Z, Y, X)
    angle_deg: rotation of the phantom around the Z axis (degrees)
    The camera is fixed looking from the +X direction — integrate along X (axis=2)
    """
    vol_rot = rotate(volume, angle_deg, axes=(1, 2),
                     reshape=False, order=1,
                     mode="constant", cval=0.0)
    proj = vol_rot.sum(axis=2)  # sum over X → side projection
    proj = proj.astype(np.float32)
    proj -= proj.min()
    if proj.max() > 0:
        proj /= proj.max()
    return proj  # 2D, shape (Z, Y) (Z, Y ordering here)

# ---------- 3. Simple noise model: Poisson noise ----------

def add_poisson_noise(proj, scale=20000):
    proj_scaled = proj * scale
    noisy_counts = np.random.poisson(proj_scaled)
    noisy = noisy_counts / scale
    noisy -= noisy.min()
    if noisy.max() > 0:
        noisy /= noisy.max()
    return noisy

# ---------- 4. Select feature pixels from a reference projection ----------

def select_feature_pixels(ref_proj, num_pixels=40, margin=5):
    """
    Select bright pixels from a reference side projection (avoid edges).
    Returns a list of (row, col) coordinates.
    """
    h, w = ref_proj.shape
    mask = np.zeros_like(ref_proj, dtype=bool)
    mask[:margin, :] = True
    mask[-margin:, :] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True

    cand = ref_proj.copy()
    cand[mask] = 0.0

    coords = []
    for _ in range(num_pixels):
        idx = np.unravel_index(np.argmax(cand), cand.shape)
        coords.append(idx)
        cand[idx] = 0.0
    return coords

# ---------- 5. Build an angle -> noiseless projection library ----------

def build_projection_library(volume, angles):
    """
    返回 dict: angle_deg -> projection (无噪声)
    """
    lib = {}
    for ang in angles:
        lib[ang] = project_side(volume, ang)
    return lib

# ---------- 6. Estimate angle using the library & feature pixels ----------

def estimate_angle_from_library(meas_proj,
                                lib,
                                candidate_angles,
                                sample_pixels):
    """
    meas_proj: measured noisy projection
    lib: {angle: noiseless_proj}
    candidate_angles: search only within this set of angles
    sample_pixels: [(r,c), ...] selected feature pixel coordinates
    """
    best_ang = None
    best_mse = float("inf")

    for ang in candidate_angles:
        ref = lib[ang]
        diffs = []
        for (r, c) in sample_pixels:
            diffs.append((ref[r, c] - meas_proj[r, c]) ** 2)
        mse = float(np.mean(diffs))
        if mse < best_mse:
            best_mse = mse
            best_ang = ang

    return best_ang, best_mse

# ---------- 7. Full demo: one reference + one measurement ----------

def run_two_side_demo(volume,
                      lib,
                      feature_angle=0.0,
                      theta_cmd=30.0,
                      noise_std_deg=1.0,
                      search_half_range=10.0,
                      save_images: bool = False,
                      save_dir: str = "projections_noisy"):
    """
    feature_angle: which angle's noiseless side projection to use for selecting features (usually 0°)
    theta_cmd: commanded angle for the motor/turntable
    noise_std_deg: std dev of angle noise in degrees
    save_images: if True, save proj_true and proj_meas as PNGs under `save_dir`
    save_dir: directory to save images into (created if missing)
    """
    # 1) Select feature pixels on the projection at the reference angle
    ref_for_features = lib[feature_angle]
    sample_pixels = select_feature_pixels(ref_for_features,
                                          num_pixels=40,
                                          margin=5)

    # 2) Generate true angle & noisy measurement
    theta_true = theta_cmd + np.random.normal(0.0, noise_std_deg)
    proj_true = project_side(volume, theta_true)
    proj_meas = add_poisson_noise(proj_true)

    # optionally save the true/noisy projection images
    if save_images:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        # scale to 0-255 u8
        def to_u8(img):
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return img_u8

        true_u8 = to_u8(proj_true)
        meas_u8 = to_u8(proj_meas)

        # filenames include commanded and true angles for traceability
        out_true = out / f"proj_true_cmd{int(theta_cmd):03d}_true{theta_true:.2f}.png"
        out_meas = out / f"proj_meas_cmd{int(theta_cmd):03d}_true{theta_true:.2f}.png"
        imageio.imwrite(out_true, true_u8)
        imageio.imwrite(out_meas, meas_u8)
        print(f"Saved true proj: {out_true}")
        print(f"Saved measured (noisy) proj: {out_meas}")

    # 3) Search the library around theta_cmd (discrete angles)
    all_angles = sorted(lib.keys())           # e.g. 0,5,10,...,355
    # Restrict candidates to within cmd ± search_half_range
    candidate_angles = [a for a in all_angles
                        if abs(((a - theta_cmd + 180) % 360) - 180) <= search_half_range]

    theta_est, mse = estimate_angle_from_library(
        meas_proj=proj_meas,
        lib=lib,
        candidate_angles=candidate_angles,
        sample_pixels=sample_pixels,
    )

    print("====================================")
    print(f"θ_cmd  (commanded) = {theta_cmd:.2f} deg")
    print(f"θ_true (actual)    = {theta_true:.2f} deg")
    print(f"θ_est  (estimated) = {theta_est:.2f} deg")
    print(f"error (est-true)   = {theta_est - theta_true:.2f} deg")
    print(f"MSE on features    = {mse:.4e}")
    print(f"search range       = ±{search_half_range:.1f} deg around θ_cmd")
    return theta_true, theta_est

def main():
    header_path = "Activity_Jas.h33"
    vol = load_i33_volume(header_path)
    print("phantom volume shape:", vol.shape)

    # Pre-build an angle -> projection library (noiseless)
    angle_step = 5.0
    lib_angles = np.arange(0.0, 360.0, angle_step)
    print("building projection library...")
    lib = build_projection_library(vol, lib_angles)
    print("library size:", len(lib))

    # 跑几个不同 command angle 的 demo
    for theta_cmd in [0.0, 20.0, 45.0, 90.0]:
        run_two_side_demo(
            volume=vol,
            lib=lib,
            feature_angle=0.0,
            theta_cmd=theta_cmd,
            noise_std_deg=1.0,
            search_half_range=10.0,
            save_images=True,
            save_dir="projections_noisy",
        )

if __name__ == "__main__":
    main()


#所以整个流程是：
#	1.	读入 3D phantom（Activity_Jas.h33 + .i33）
#	2.	定义一个 side-view 投影模型：phantom 绕 Z 轴旋转，相机固定在 +X 方向；
#	3.	在每个角度生成一张无噪声侧视投影 → 形成 angle → projection 的“字典库”
#	4.	给定一个“指令角度 θ_cmd”，往上加一个小的随机机械误差 → 得到真实角度 θ_true
#	5.	用 θ_true 生成真实无噪声投影，再加 Poisson 噪声 → 模拟一次测量
#	6.	在某张参考投影上选一批“亮点像素”当特征；
#	7.	用这批特征像素，把 noisy 投影和库里各个角度的投影做 MSE 对比 → 找最像的 → 得到 θ_est
#	8.	打印 θ_cmd / θ_true / θ_est / 误差 / MSE。