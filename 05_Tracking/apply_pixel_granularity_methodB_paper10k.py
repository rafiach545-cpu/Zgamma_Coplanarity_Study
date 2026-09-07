import numpy as np
import pandas as pd

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_eta18.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_eta18_pixelized.csv"

df = pd.read_csv(INPUT)

# ATLAS Run-2 pixel dimensions:
# IBL:       50 um (r-phi) x 250 um (z)
# Pixel B0-2:50 um (r-phi) x 400 um (z)
#
# Knapen et al. state that pixel SIZE, rather than intrinsic
# resolution, is used for the pixel detector.
PIXEL_PITCH_RPHI_MM = {
    1: 0.050,
    2: 0.050,
    3: 0.050,
    4: 0.050,
}

PIXEL_PITCH_Z_MM = {
    1: 0.250,   # IBL
    2: 0.400,
    3: 0.400,
    4: 0.400,
}

out = df.copy()

pixel_mask = out["layer_id"].isin([1, 2, 3, 4])

for idx in out.index[pixel_mask]:

    layer = int(out.at[idx, "layer_id"])
    R = float(out.at[idx, "r"])
    phi = float(out.at[idx, "phi"])
    z = float(out.at[idx, "z"])

    pitch_s = PIXEL_PITCH_RPHI_MM[layer]
    pitch_z = PIXEL_PITCH_Z_MM[layer]

    # Local coordinate along cylinder circumference
    s = R * phi

    # Replace true crossing by center of nearest pixel cell
    s_center = np.round(s / pitch_s) * pitch_s
    z_center = np.round(z / pitch_z) * pitch_z

    phi_center = s_center / R

    out.at[idx, "phi"] = phi_center
    out.at[idx, "z"] = z_center
    out.at[idx, "x"] = R * np.cos(phi_center)
    out.at[idx, "y"] = R * np.sin(phi_center)

out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Total hits:", len(out))
print("Pixel hits modified:", pixel_mask.sum())
print("SCT hits unchanged:", (~pixel_mask).sum())

print()
print("Example pixel hits before/after:")
cols = ["event", "charge", "layer_id", "x", "y", "z", "phi"]

before = df.loc[pixel_mask, cols].head(5).copy()
after  = out.loc[pixel_mask, cols].head(5).copy()

print("\nBEFORE:")
print(before.to_string(index=False))

print("\nAFTER:")
print(after.to_string(index=False))
