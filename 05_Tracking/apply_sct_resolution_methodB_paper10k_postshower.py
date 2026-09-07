import numpy as np
import pandas as pd

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_postshower_eta18_pixelized.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_postshower_eta18_detector_response.csv"

df = pd.read_csv(INPUT)
out = df.copy()

# Knapen et al.: use twice the intrinsic SCT accuracy.
# ATLAS SCT barrel intrinsic accuracies:
#   R-phi ~ 17 um
#   z     ~ 580 um
#
# Therefore reproduction uses:
SIGMA_RPHI_MM = 2.0 * 17.0 / 1000.0    # 0.034 mm
SIGMA_Z_MM    = 2.0 * 580.0 / 1000.0   # 1.16 mm

rng = np.random.default_rng(42)

sct_mask = out["layer_id"].isin([5, 6, 7, 8])

for idx in out.index[sct_mask]:

    R = float(out.at[idx, "r"])
    phi = float(out.at[idx, "phi"])
    z = float(out.at[idx, "z"])

    # Local transverse coordinate along cylinder circumference
    s = R * phi

    s_meas = s + rng.normal(0.0, SIGMA_RPHI_MM)
    z_meas = z + rng.normal(0.0, SIGMA_Z_MM)

    phi_meas = s_meas / R

    out.at[idx, "phi"] = phi_meas
    out.at[idx, "z"] = z_meas
    out.at[idx, "x"] = R * np.cos(phi_meas)
    out.at[idx, "y"] = R * np.sin(phi_meas)

out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Total hits:", len(out))
print("Pixel hits unchanged:", (~sct_mask).sum())
print("SCT hits smeared:", sct_mask.sum())

print()
print("SCT sigma Rphi:", SIGMA_RPHI_MM, "mm")
print("SCT sigma z:", SIGMA_Z_MM, "mm")

cols = ["event", "charge", "layer_id", "x", "y", "z", "phi"]

print("\nBEFORE:")
print(df.loc[sct_mask, cols].head(5).to_string(index=False))

print("\nAFTER:")
print(out.loc[sct_mask, cols].head(5).to_string(index=False))
