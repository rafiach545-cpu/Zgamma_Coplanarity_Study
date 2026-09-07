import sys
import os
import numpy as np
import pandas as pd

sys.path.append("00_Shared_Code")
sys.path.append("05_Tracking")

from charged_trajectory import simulate_charged_track
from simplified_atlas_geometry import (
    ATLAS_LAYER_RADII_MM,
    ATLAS_LAYER_ZMAX_MM,
    cylinder_segment_intersection,
)

INPUT = "02_Pythia/minbias_1000.csv"

OUTPUT = "05_Tracking/minbias_1000_hits_cylinders_vertices.csv"
CHECKPOINT = "05_Tracking/minbias_1000_hits_vertices_checkpoint.csv"
STATE = "05_Tracking/minbias_1000_hits_vertices_checkpoint_state.txt"

CHUNK = 1000


def first_chronological_crossings(traj_xyz):
    """
    Follow trajectory in time order.

    Keep the first physical crossing of each cylindrical layer,
    regardless of whether the segment is moving inward or outward.

    Maximum one hit per detector layer.
    """

    xyz = np.asarray(traj_xyz, dtype=np.float64)

    if len(xyz) < 2:
        return []

    found_layers = set()
    hits = []

    # Scan trajectory chronologically
    for i in range(len(xyz) - 1):

        A = xyz[i]
        B = xyz[i + 1]

        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue

        segment_hits = []

        for layer_id, radius_mm in ATLAS_LAYER_RADII_MM.items():

            if layer_id in found_layers:
                continue

            crossings = cylinder_segment_intersection(
                A,
                B,
                radius_mm,
            )

            for t, point in crossings:

                z_max = ATLAS_LAYER_ZMAX_MM.get(layer_id)

                if z_max is not None and abs(point[2]) > z_max:
                    continue

                segment_hits.append(
                    (
                        float(t),
                        int(layer_id),
                        point,
                    )
                )

        # If one integration step crosses multiple cylinders,
        # retain their true time ordering inside that segment.
        segment_hits.sort(key=lambda x: x[0])

        for t, layer_id, point in segment_hits:

            if layer_id in found_layers:
                continue

            hits.append({
                "layer_id": layer_id,
                "trajectory_step": int(i),
                "t": float(t),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "r": float(np.hypot(point[0], point[1])),
                "phi": float(np.arctan2(point[1], point[0])),
            })

            found_layers.add(layer_id)

        if len(found_layers) == len(ATLAS_LAYER_RADII_MM):
            break

    return hits


df = pd.read_csv(INPUT)
df = df[df["eta"].abs() < 1.8].copy()

# Preserve original row index as stable particle ID
df = df.reset_index().rename(columns={"index": "original_index"})

start = 0

if os.path.exists(STATE):
    with open(STATE) as f:
        txt = f.read().strip()
        if txt:
            start = int(txt)

print("Input barrel charged particles:", len(df))
print("Input min-bias events:", df["event"].nunique())
print("Resuming from particle:", start)

for chunk_start in range(start, len(df), CHUNK):

    chunk_end = min(chunk_start + CHUNK, len(df))
    all_hits = []

    for pos in range(chunk_start, chunk_end):

        p = df.iloc[pos]

        xyz = simulate_charged_track(
            origin=np.array(
                [
                    p["xProd"],
                    p["yProd"],
                    p["zProd"],
                ],
                dtype=float,
            ),
            momentum=np.array(
                [
                    p["px"],
                    p["py"],
                    p["pz"],
                ],
                dtype=float,
            ),
            charge=int(p["charge"]),
            mass=float(p["mass"]),
            b_field=2.0,
            n_steps=3000,
            t_max=10.0,
        )

        hits = first_chronological_crossings(xyz)

        for h in hits:
            all_hits.append({
                "pileup_event": int(p["event"]),
                "particle_id": int(p["original_index"]),
                "pdg_id": int(p["id"]),
                "charge": float(p["charge"]),
                "mass": float(p["mass"]),
                "pT": float(p["pT"]),
                "eta": float(p["eta"]),
                "xProd": float(p["xProd"]),
                "yProd": float(p["yProd"]),
                "zProd": float(p["zProd"]),
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
        index=False,
    )

    with open(STATE, "w") as f:
        f.write(str(chunk_end))

    print(
        f"Checkpoint saved: "
        f"{chunk_end}/{len(df)} particles"
    )

# Build final output from checkpoint
final_df = pd.read_csv(CHECKPOINT)
final_df.to_csv(OUTPUT, index=False)

print("\n=== FINAL DISPLACED-VERTEX SAMPLE ===")
print("Output:", OUTPUT)
print("Total hits:", len(final_df))
print(
    "Pile-up events with hits:",
    final_df["pileup_event"].nunique(),
)
print(
    "Particles with hits:",
    final_df["particle_id"].nunique(),
)

counts = final_df.groupby("particle_id").size()

print("Min hits/particle:", counts.min())
print("Max hits/particle:", counts.max())

print("\nHits by layer:")
print(
    final_df["layer_id"]
    .value_counts()
    .sort_index()
)
