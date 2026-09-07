from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

input_path = ROOT / "03_Photon_Selection" / "leading_isolated_photons_10000.csv"
output_path = ROOT / "04_Conversion" / "converted_pairs_lowpt_debug.csv"

photons = pd.read_csv(input_path)

# Dedicated diagnostic sample: ALL photons with pT < 5 GeV
photons = photons[photons["pT"] < 5].copy()

r_conv_mm = 23.5   # ATLAS ID beam-pipe inner radius, arXiv:1803.00844

rows = []

for _, row in photons.iterrows():

    pT = row["pT"]
    eta = row["eta"]
    phi = row["phi"]

    E = pT * np.cosh(eta)

    px = pT * np.cos(phi)
    py = pT * np.sin(phi)
    pz = pT * np.sinh(eta)

    scale = r_conv_mm / pT

    x = scale * px
    y = scale * py
    z = scale * pz

    for particle, charge in [("electron", -1), ("positron", +1)]:

        rows.append({
            "event": row["event"],
            "particle": particle,
            "charge": charge,
            "x": x,
            "y": y,
            "z": z,
            "px": px / 2,
            "py": py / 2,
            "pz": pz / 2,
            "E": E / 2,
            "photon_pT": pT
        })

out = pd.DataFrame(rows)
out.to_csv(output_path, index=False)

print("Low-pT photons:", len(photons))
print("Saved particles:", len(out))
print("Saved:", output_path)
