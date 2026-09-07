import sys
import numpy as np
import pandas as pd

sys.path.append("00_Shared_Code")
sys.path.append("05_Tracking")

from charged_trajectory import simulate_charged_track, keep_outward_branch
from simplified_atlas_geometry import intersect_trajectory_with_cylinders


INPUT = "04_Conversion/converted_pairs_weighted.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders.csv"


pairs = pd.read_csv(INPUT)

print("Input particles:", len(pairs))
print("Input conversions:", pairs["event"].nunique())

all_hits = []

for idx, p in pairs.iterrows():

    xyz = simulate_charged_track(
        origin=np.array(
            [p["x"], p["y"], p["z"]],
            dtype=float
        ),
        momentum=np.array(
            [p["px"], p["py"], p["pz"]],
            dtype=float
        ),
        charge=int(p["charge"]),
        mass=0.000511,
        b_field=2.0,
        n_steps=3000,
        t_max=10.0,
    )

    xyz = keep_outward_branch(xyz)

    hits = intersect_trajectory_with_cylinders(xyz)

    for h in hits:
        all_hits.append({
            "event": p["event"],
            "charge": p["charge"],
            "layer_id": h["layer_id"],
            "trajectory_step": h["trajectory_step"],
            "x": h["x"],
            "y": h["y"],
            "z": h["z"],
            "r": h["r"],
            "phi": h["phi"],
            "photon_pT": p["photon_pT"],
            "weight": p["weight"],
        })

    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(pairs)} particles")


out = pd.DataFrame(all_hits)

out.to_csv(OUTPUT, index=False)

print()
print("Saved:", OUTPUT)
print("Total hits:", len(out))
print("Events with at least one hit:", out["event"].nunique())
print("Particles processed:", len(pairs))
print("Max hits in one event/charge:",
      out.groupby(["event", "charge"]).size().max())
