import pandas as pd
import numpy as np

from charged_trajectory import (
    simulate_charged_track,
    keep_outward_branch
)
from quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules
)

pairs_path = "/mnt/c/Users/HP/Downloads/MG5_aMC_v3_7_2/Zgamma_test/Events/run_01/converted_pairs_10000.csv"

detector_path = "/home/rafia/TrackML-note/QuirkTracking-ML/trackml_Dataset_Detector_WM/trackml_raw/detectors.csv"


pairs = pd.read_csv(pairs_path)

detector_df = pd.read_csv(detector_path)

modules = build_detector_module_table(detector_df)

print("Detector modules:", len(modules))


all_hits=[]


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


        if particle.particle == "positron":
            charge = +1
        else:
            charge = -1


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

            all_hits.append(hits)



if len(all_hits):

    hits_df = pd.concat(
        all_hits,
        ignore_index=True
    )

else:
    hits_df = pd.DataFrame()


hits_df.to_csv(
    "converted_photon_hits_curved.csv",
    index=False
)


print()
print("Saved converted_photon_hits_curved.csv")
print("Total hits:", len(hits_df))

if len(hits_df):
    print(hits_df.groupby(
        ["event","particle"]
    ).size())
