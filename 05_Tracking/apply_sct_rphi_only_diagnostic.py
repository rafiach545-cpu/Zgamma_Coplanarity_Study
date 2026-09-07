import numpy as np
import pandas as pd

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_pixelized.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_sct_rphi_only.csv"

df = pd.read_csv(INPUT)
out = df.copy()

SIGMA_RPHI_MM = 0.034

rng = np.random.default_rng(42)

mask = out["layer_id"].isin([5, 6, 7, 8])

for idx in out.index[mask]:

    R = float(out.at[idx, "r"])
    phi = float(out.at[idx, "phi"])

    s = R * phi
    s_meas = s + rng.normal(0.0, SIGMA_RPHI_MM)

    phi_meas = s_meas / R

    out.at[idx, "phi"] = phi_meas
    out.at[idx, "x"] = R * np.cos(phi_meas)
    out.at[idx, "y"] = R * np.sin(phi_meas)

    # z deliberately left unchanged for this diagnostic

out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Total hits:", len(out))
print("SCT hits modified in Rphi only:", mask.sum())
