import numpy as np
import pandas as pd

photons = pd.read_csv(
    "03_Photon_Selection/leading_isolated_photons_10000.csv"
)

# Four-momentum
photons["E"] = photons.pT * np.cosh(photons.eta)
photons["px"] = photons.pT * np.cos(photons.phi)
photons["py"] = photons.pT * np.sin(photons.phi)
photons["pz"] = photons.pT * np.sinh(photons.eta)

# Physical-like conversion weight
conversion_weight = 0.01

# Simplified fixed conversion radius
r_conv_mm = 23.5

# IMPORTANT:
# Convert ALL selected photons.
# No random 1% or 10% sampling.
converting = photons.copy()

print(
    f"Photons converted in simulation: {len(converting)} / {len(photons)}"
)

pairs = []

for _, ph in converting.iterrows():

    scale = r_conv_mm / ph.pT

    cx = ph.px * scale
    cy = ph.py * scale
    cz = ph.pz * scale

    E_each = ph.E / 2.0

    for particle, charge in [
        ("positron", +1),
        ("electron", -1)
    ]:

        pairs.append({
            "event": ph.event,
            "particle": particle,
            "charge": charge,

            "x": cx,
            "y": cy,
            "z": cz,

            "px": ph.px / 2.0,
            "py": ph.py / 2.0,
            "pz": ph.pz / 2.0,

            "E": E_each,

            "photon_pT": ph.pT,

            # same physical weight carried by both particles
            "weight": conversion_weight
        })

out = pd.DataFrame(pairs)

out.to_csv(
    "04_Conversion/converted_pairs_weighted.csv",
    index=False
)

n_conversions = len(out) // 2

print(
    f"Saved {len(out)} particles "
    f"({n_conversions} simulated conversions)"
)

print(
    f"Effective physical converted-photon count "
    f"= {n_conversions * conversion_weight:.2f}"
)
