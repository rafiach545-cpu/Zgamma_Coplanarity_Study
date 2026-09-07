import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_10_hits_detector_response.csv"

OUT = "06_Coplanarity/smoke_overlay_1hard_5pu.csv"

PIXEL_PITCH_RPHI_MM = {
    1: 0.050,
    2: 0.050,
    3: 0.050,
    4: 0.050,
}

PIXEL_PITCH_Z_MM = {
    1: 0.250,
    2: 0.400,
    3: 0.400,
    4: 0.400,
}

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

hard_event = sorted(hard["event"].unique())[0]
h = hard[hard["event"] == hard_event].copy()

h["source"] = "hard"
h["source_event"] = hard_event

pu_events = sorted(pu["pileup_event"].unique())[:5]
p = pu[pu["pileup_event"].isin(pu_events)].copy()

p["source"] = "pileup"
p["source_event"] = p["pileup_event"]

keep = [
    "source",
    "source_event",
    "layer_id",
    "trajectory_step",
    "x",
    "y",
    "z",
    "r",
    "phi",
]

combo = pd.concat(
    [h[keep], p[keep]],
    ignore_index=True
)

print("Hard event:", hard_event)
print("Pile-up interactions:", pu_events)

print("\nHits before joint merging:", len(combo))
print("Hard hits:", (combo["source"] == "hard").sum())
print("Pile-up hits:", (combo["source"] == "pileup").sum())

print("\nBefore merge by layer:")
print(combo["layer_id"].value_counts().sort_index())

pixel = combo[
    combo["layer_id"].isin([1,2,3,4])
].copy()

sct = combo[
    combo["layer_id"].isin([5,6,7,8])
].copy()

merged_rows = []

for layer, g in pixel.groupby("layer_id", sort=False):

    layer = int(layer)
    R = float(g["r"].iloc[0])

    pitch_s = PIXEL_PITCH_RPHI_MM[layer]
    pitch_z = PIXEL_PITCH_Z_MM[layer]

    g = g.copy()

    g["pix_s"] = np.rint(
        (R * g["phi"]) / pitch_s
    ).astype(int)

    g["pix_z"] = np.rint(
        g["z"] / pitch_z
    ).astype(int)

    rows = g.to_dict("records")
    used = [False] * len(rows)

    for i in range(len(rows)):

        if used[i]:
            continue

        cluster = [i]
        used[i] = True

        changed = True

        while changed:
            changed = False

            for j in range(len(rows)):

                if used[j]:
                    continue

                for k in cluster:

                    ds = abs(
                        rows[j]["pix_s"] -
                        rows[k]["pix_s"]
                    )

                    dz = abs(
                        rows[j]["pix_z"] -
                        rows[k]["pix_z"]
                    )

                    if ds <= 1 and dz <= 1:
                        cluster.append(j)
                        used[j] = True
                        changed = True
                        break

        members = [rows[k] for k in cluster]

        x = np.mean([m["x"] for m in members])
        y = np.mean([m["y"] for m in members])
        z = np.mean([m["z"] for m in members])

        sources = sorted(set(m["source"] for m in members))

        hard_members = sum(
            m["source"] == "hard"
            for m in members
        )

        pileup_members = sum(
            m["source"] == "pileup"
            for m in members
        )

        merged_rows.append({
            "source": "+".join(sources),
            "source_event": -1,
            "layer_id": layer,
            "trajectory_step":
                min(m["trajectory_step"] for m in members),
            "x": x,
            "y": y,
            "z": z,
            "r": R,
            "phi": np.arctan2(y, x),
            "n_merged": len(members),
            "hard_members": hard_members,
            "pileup_members": pileup_members,
        })

pixel_merged = pd.DataFrame(merged_rows)

sct = sct.copy()
sct["n_merged"] = 1
sct["hard_members"] = (sct["source"] == "hard").astype(int)
sct["pileup_members"] = (sct["source"] == "pileup").astype(int)

final = pd.concat(
    [pixel_merged, sct],
    ignore_index=True,
    sort=False
)

final = final.sort_values(
    ["layer_id", "trajectory_step"]
).reset_index(drop=True)

final.to_csv(OUT, index=False)

print("\n===== JOINT MERGING =====")

print("Hits after joint merging:", len(final))
print("Removed:", len(combo) - len(final))

print(
    "Pixel clusters with >1 hit:",
    int((pixel_merged["n_merged"] > 1).sum())
)

print(
    "Largest pixel cluster:",
    int(pixel_merged["n_merged"].max())
)

mixed = pixel_merged[
    (pixel_merged["hard_members"] > 0) &
    (pixel_merged["pileup_members"] > 0)
]

print(
    "Hard-pileup mixed pixel clusters:",
    len(mixed)
)

print(
    "Hard hits absorbed into mixed clusters:",
    int(mixed["hard_members"].sum())
)

print("\nAfter merge by layer:")
print(
    final["layer_id"]
         .value_counts()
         .sort_index()
)

xyz = final[["x","y","z"]].to_numpy(float)

layers = np.array(
    [MAP[int(x)] for x in final["layer_id"]],
    dtype=int
)

result = paper_plane_finding_exact8(
    xyz,
    layers
)

print("\n===== PLANE FINDER =====")
print("Found:", result.get("found", False))

for key in [
    "delta_s",
    "delta_w",
    "reason",
]:
    if key in result:
        print(key + ":", result[key])

print("\nSaved:", OUT)
