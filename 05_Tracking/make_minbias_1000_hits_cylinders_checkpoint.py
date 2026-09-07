import sys
import os
import numpy as np
import pandas as pd

sys.path.append("00_Shared_Code")
sys.path.append("05_Tracking")

from charged_trajectory import simulate_charged_track, keep_outward_branch
from simplified_atlas_geometry import intersect_trajectory_with_cylinders

INPUT = "02_Pythia/minbias_1000.csv"
OUTPUT = "05_Tracking/minbias_1000_hits_cylinders.csv"
CHECKPOINT = "05_Tracking/minbias_1000_hits_checkpoint.csv"
STATE = "05_Tracking/minbias_1000_hits_checkpoint_state.txt"

CHUNK = 1000

df = pd.read_csv(INPUT)
df = df[df["eta"].abs() < 1.8].copy()

# Make stable sequential processing index
df = df.reset_index().rename(columns={"index": "original_index"})

start = 0

if os.path.exists(STATE):
    with open(STATE) as f:
        start = int(f.read().strip())

print("Input barrel charged particles:", len(df))
print("Input min-bias events:", df["event"].nunique())
print("Resuming from particle:", start)

for chunk_start in range(start, len(df), CHUNK):

    chunk_end = min(chunk_start + CHUNK, len(df))

    all_hits = []

    for pos in range(chunk_start, chunk_end):

        p = df.iloc[pos]

        xyz = simulate_charged_track(
            origin=np.array([0.0, 0.0, 0.0]),
            momentum=np.array(
                [p["px"], p["py"], p["pz"]],
                dtype=float
            ),
            charge=int(p["charge"]),
            mass=float(p["mass"]),
            b_field=2.0,
            n_steps=3000,
            t_max=10.0,
        )

        xyz = keep_outward_branch(xyz)

        hits = intersect_trajectory_with_cylinders(xyz)

        for h in hits:
            all_hits.append({
                "pileup_event": int(p["event"]),
                "particle_id": int(p["original_index"]),
                "pdg_id": int(p["id"]),
                "charge": float(p["charge"]),
                "mass": float(p["mass"]),
                "pT": float(p["pT"]),
                "eta": float(p["eta"]),
                "layer_id": int(h["layer_id"]),
                "trajectory_step": int(h["trajectory_step"]),
                "x": float(h["x"]),
                "y": float(h["y"]),
                "z": float(h["z"]),
                "r": float(h["r"]),
                "phi": float(h["phi"]),
            })

    chunk_df = pd.DataFrame(all_hits)

    header = not os.path.exists(CHECKPOINT)

    chunk_df.to_csv(
        CHECKPOINT,
        mode="a",
        header=header,
        index=False
    )

    with open(STATE, "w") as f:
        f.write(str(chunk_end))

    print(
        f"Checkpoint saved: "
        f"{chunk_end}/{len(df)} particles"
    )

print("\nAll particles processed.")

out = pd.read_csv(CHECKPOINT)

out.to_csv(
    OUTPUT,
    index=False
)

print("Saved final:", OUTPUT)
print("Total hits:", len(out))
print(
    "Pile-up events with hits:",
    out["pileup_event"].nunique()
)
print(
    "Particles with hits:",
    out["particle_id"].nunique()
)

counts = out.groupby("particle_id").size()

print("Min hits/particle:", counts.min())
print("Max hits/particle:", counts.max())

print("\nHits by layer:")
print(
    out["layer_id"]
       .value_counts()
       .sort_index()
)
