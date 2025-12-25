import numpy as np
from pathlib import Path
from scipy.ndimage import rotate
import imageio.v2 as imageio

# ---------- 1. Read .h33 header ----------
def load_h33_header(header_path):
    header = {}
    with open(header_path, "r") as f:
        for line in f:
            if ":=" not in line:
                continue
            key, val = line.split(":=", 1)
            key = key.strip()
            val = val.strip()
            # remove leading exclamation mark '!' from keys
            if key.startswith("!"):
                key = key[1:].strip()
            header[key] = val
    return header

# ---------- 2. Load voxel volume (.i33) ----------
def load_i33_volume(header_path):
    header = load_h33_header(header_path)


    nx = int(header["matrix size [1]"])
    ny = int(header["matrix size [2]"])
    nz = int(header["number of slices"])

    data_file = header["name of data file"]
    data_path = Path(header_path).with_name(data_file)

    print(">>> Attempting to read voxel file:", data_path.name)

    # According to header: unsigned integer, 2 bytes -> uint16
    arr = np.fromfile(data_path, dtype=np.uint16)
    expected = nx * ny * nz
    if arr.size != expected:
        raise RuntimeError(f"Data length mismatch: read {arr.size} pixels, expected {expected}")

    # Z, Y, X
    vol = arr.reshape((nz, ny, nx))
    return vol

# ---------- 3. Parallel-beam projection ----------
def project_parallel(volume, angle_deg):
    # Rotate around Z axis (in the (Y, X) plane)
    vol_rot = rotate(volume, angle_deg, axes=(1, 2),
                     reshape=False, order=1,
                     mode="constant", cval=0.0)
    # Integrate along X (sum over X axis)
    proj = vol_rot.sum(axis=2)
    proj = proj - proj.min()
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def main():
    header_path = "Activity_Jas.h33"

    print(">>> Loading 3D phantom...")
    vol = load_i33_volume(header_path)
    print("Voxel dimensions (Z, Y, X):", vol.shape)

    out = Path("projections")
    out.mkdir(exist_ok=True)

    angles = range(0, 360, 5)
    print(">>> Starting projection generation, total", len(angles), "angles")

    for ang in angles:
        proj = project_parallel(vol, ang)
        proj_u8 = (proj * 255).astype(np.uint8)
        out_path = out / f"proj_{ang:03d}.png"
    imageio.imwrite(out_path, proj_u8)
    print("Saved:", out_path)

    print(">>> Done! All projections are saved in projections/.")

if __name__ == "__main__":
    main()

# Construct 3D image
# For each ray along X, integrate the phantom to obtain a 2D image.
# The two axes of the image are Y and Z.