import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

photons = pd.read_csv(
    "../03_Photon_Selection/leading_isolated_photons_10000.csv"
)

# four momentum
photons["E"] = photons.pT * np.cosh(photons.eta)
photons["px"] = photons.pT * np.cos(photons.phi)
photons["py"] = photons.pT * np.sin(photons.phi)
photons["pz"] = photons.pT * np.sinh(photons.eta)


# Statistical sample only
conversion_prob = 0.10

r_conv_mm = 23.5   # ATLAS ID beam-pipe inner radius, arXiv:1803.00844


mask = rng.random(len(photons)) < conversion_prob

converting = photons[mask].copy()

print(
    f"Converted photons: {len(converting)} / {len(photons)}"
)


pairs = []


for _, ph in converting.iterrows():

    scale = r_conv_mm / ph.pT

    cx = ph.px * scale
    cy = ph.py * scale
    cz = ph.pz * scale

    E_each = ph.E / 2


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

            "px": ph.px/2,
            "py": ph.py/2,
            "pz": ph.pz/2,

            "E": E_each,

            "photon_pT": ph.pT

        })


out = pd.DataFrame(pairs)


out.to_csv(
    "converted_pairs_highstat.csv",
    index=False
)


print(
    f"Saved {len(out)} particles "
    f"({len(out)//2} conversions)"
)
