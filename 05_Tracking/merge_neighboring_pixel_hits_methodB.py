import numpy as np
import pandas as pd

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_final_response.csv"

df = pd.read_csv(INPUT)

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

pixel = df[df["layer_id"].isin([1,2,3,4])].copy()
sct   = df[df["layer_id"].isin([5,6,7,8])].copy()

merged_rows = []

for (event, layer), g in pixel.groupby(["event", "layer_id"], sort=False):

    layer = int(layer)
    R = float(g["r"].iloc[0])

    pitch_s = PIXEL_PITCH_RPHI_MM[layer]
    pitch_z = PIXEL_PITCH_Z_MM[layer]

    g = g.copy()

    # Recover discrete pixel-cell indices from already pixelized coordinates
    g["pix_s"] = np.rint((R * g["phi"]) / pitch_s).astype(int)
    g["pix_z"] = np.rint(g["z"] / pitch_z).astype(int)

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
                    ds = abs(rows[j]["pix_s"] - rows[k]["pix_s"])
                    dz = abs(rows[j]["pix_z"] - rows[k]["pix_z"])

                    if ds <= 1 and dz <= 1:
                        cluster.append(j)
                        used[j] = True
                        changed = True
                        break

        members = [rows[k] for k in cluster]

        # Cluster position = mean of pixel-centre positions
        x = np.mean([m["x"] for m in members])
        y = np.mean([m["y"] for m in members])
        z = np.mean([m["z"] for m in members])

        phi = np.arctan2(y, x)

        merged_rows.append({
            "event": event,
            "charge": members[0]["charge"] if len(members) == 1 else 0,
            "layer_id": layer,
            "trajectory_step": min(m["trajectory_step"] for m in members),
            "x": x,
            "y": y,
            "z": z,
            "r": R,
            "phi": phi,
            "photon_pT": members[0]["photon_pT"],
            "weight": members[0]["weight"],
            "n_merged": len(members),
        })

pixel_merged = pd.DataFrame(merged_rows)

sct["n_merged"] = 1

out = pd.concat([pixel_merged, sct], ignore_index=True)
out = out.sort_values(["event", "layer_id", "trajectory_step"]).reset_index(drop=True)

out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print()
print("Hits before merging:", len(df))
print("Hits after merging:", len(out))
print("Removed by pixel merging:", len(df) - len(out))
print()
print("Pixel hits before:", len(pixel))
print("Pixel hits after:", len(pixel_merged))
print("Merged clusters with >1 hit:", (pixel_merged["n_merged"] > 1).sum())
print("Largest pixel cluster:", pixel_merged["n_merged"].max())
print()
print("SCT hits unchanged:", len(sct))
