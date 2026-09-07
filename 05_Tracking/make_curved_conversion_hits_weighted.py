import pandas as pd
import numpy as np
import sys

sys.path.append("00_Shared_Code")

from charged_trajectory import (
    simulate_charged_track,
    keep_outward_branch
)

from quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules
)

pairs_path = "04_Conversion/converted_pairs_weighted.csv"

detector_path = "/home/rafia/TrackML-note/QuirkTracking-ML/trackml_Dataset_Detector_WM/trackml_raw/detectors.csv"

pairs = pd.read_csv(pairs_path)
detector_df = pd.read_csv(detector_path)

modules = build_detector_module_table(detector_df)

print("Detector modules:", len(modules))
print("Input particles:", len(pairs))
print("Input conversions:", len(pairs) // 2)

all_hits = []

for event, group in pairs.groupby("event"):

    for _, particle in group.iterrows():

        origin = np.array([
            particle.x,
            particle.y,
            particle.z
        ])

        momentum = np.array([
            particle.px,
            particle.py,
            particle.pz
        ])

        charge = +1 if particle.particle == "positron" else -1

        xyz = simulate_charged_track(
            origin=origin,
            momentum=momentum,
            charge=charge,
            mass=0.000511,
            b_field=2.0,
            n_steps=3000,
            t_max=10
        )

        xyz = keep_outward_branch(xyz)

        hits = intersect_track_with_modules(
            xyz,
            modules
        )

        if len(hits):

            hits["event"] = event
            hits["particle"] = particle.particle
            hits["charge"] = charge

            # carry photon information forward
            hits["photon_pT"] = particle.photon_pT
            hits["weight"] = particle.weight

            all_hits.append(hits)

if len(all_hits):

    hits_df = pd.concat(
        all_hits,
        ignore_index=True
    )

else:
    hits_df = pd.DataFrame()

output_path = "05_Tracking/converted_photon_hits_weighted_curved.csv"

hits_df.to_csv(
    output_path,
    index=False
)

print()
print("Saved:", output_path)
print("Total hits:", len(hits_df))

if len(hits_df):

    print(
        "Events with at least one detector hit:",
        hits_df["event"].nunique()
    )

    print(
        "Particles with hits:",
        hits_df[["event", "particle"]]
        .drop_duplicates()
        .shape[0]
    )
