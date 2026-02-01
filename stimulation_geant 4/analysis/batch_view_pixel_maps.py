import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# paths
# -----------------------------
MAP_DIR = "build/maps"
OUT_DIR = "analysis/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# find all pixel map csv files
csv_files = sorted(glob.glob(os.path.join(MAP_DIR, "pixel_map_angle_*.csv")))

print(f"Found {len(csv_files)} pixel maps")

# -----------------------------
# loop over angles
# -----------------------------
for csv_path in csv_files:
    df = pd.read_csv(csv_path)

    # build 64x64 image
    img = np.zeros((64, 64), dtype=np.float64)

    for _, row in df.iterrows():
        iy = int(row["iy"])
        iz = int(row["iz"])
        edep = float(row["edep_keV"])
        img[iy, iz] += edep

    # normalize (optional but recommended)
    if img.sum() > 0:
        img /= img.sum()

    angle = csv_path.split("_")[-1].replace(".csv", "")
    out_png = os.path.join(OUT_DIR, f"pixel_map_angle_{angle}.png")

    plt.figure(figsize=(4, 4))
    plt.imshow(img, origin="lower", cmap="inferno")
    plt.colorbar(label="Normalized intensity")
    plt.title(f"Angle = {angle} deg")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved {out_png}")

print("Done.")
