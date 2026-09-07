import numpy as np
import pandas as pd

INPUT = "05_Tracking/minbias_10_hits_cylinders_beamspot.csv"

PIXELIZED = "05_Tracking/minbias_10_hits_pixelized.csv"
SMEARED   = "05_Tracking/minbias_10_hits_detector_response.csv"
FINAL     = "05_Tracking/minbias_10_hits_final_response.csv"

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

SIGMA_RPHI_MM = 0.034
SIGMA_Z_MM = 1.16

df = pd.read_csv(INPUT)

# ============================================================
# 1. PIXEL GRANULARITY
# ============================================================

pix = df.copy()

pixel_mask = pix["layer_id"].isin([1,2,3,4])

for idx in pix.index[pixel_mask]:

    layer = int(pix.at[idx, "layer_id"])
    R = float(pix.at[idx, "r"])
    phi = float(pix.at[idx, "phi"])
    z = float(pix.at[idx, "z"])

    pitch_s = PIXEL_PITCH_RPHI_MM[layer]
    pitch_z = PIXEL_PITCH_Z_MM[layer]

    s = R * phi

    s_center = np.round(s / pitch_s) * pitch_s
    z_center = np.round(z / pitch_z) * pitch_z

    phi_center = s_center / R

    pix.at[idx, "phi"] = phi_center
    pix.at[idx, "z"] = z_center
    pix.at[idx, "x"] = R * np.cos(phi_center)
    pix.at[idx, "y"] = R * np.sin(phi_center)

pix.to_csv(PIXELIZED, index=False)

print("Saved:", PIXELIZED)
print("Pixel hits quantized:", int(pixel_mask.sum()))


# ============================================================
# 2. SCT RESPONSE
# ============================================================

resp = pix.copy()

rng = np.random.default_rng(42)

sct_mask = resp["layer_id"].isin([5,6,7,8])

for idx in resp.index[sct_mask]:

    R = float(resp.at[idx, "r"])
    phi = float(resp.at[idx, "phi"])
    z = float(resp.at[idx, "z"])

    s = R * phi

    s_meas = s + rng.normal(0.0, SIGMA_RPHI_MM)
    z_meas = z + rng.normal(0.0, SIGMA_Z_MM)

    phi_meas = s_meas / R

    resp.at[idx, "phi"] = phi_meas
    resp.at[idx, "z"] = z_meas
    resp.at[idx, "x"] = R * np.cos(phi_meas)
    resp.at[idx, "y"] = R * np.sin(phi_meas)

resp.to_csv(SMEARED, index=False)

print("Saved:", SMEARED)
print("SCT hits smeared:", int(sct_mask.sum()))


# ============================================================
# 3. PIXEL MERGING
# ============================================================

pixel = resp[resp["layer_id"].isin([1,2,3,4])].copy()
sct = resp[resp["layer_id"].isin([5,6,7,8])].copy()

merged_rows = []

for (pev, layer), g in pixel.groupby(
    ["pileup_event", "layer_id"],
    sort=False
):

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

        phi = np.arctan2(y, x)

        first = members[0]

        merged_rows.append({
            "pileup_event": int(pev),

            # only meaningful for unmerged single-particle hit
            "particle_id":
                int(first["particle_id"])
                if len(members) == 1 else -1,

            "pdg_id":
                int(first["pdg_id"])
                if len(members) == 1 else 0,

            "charge":
                float(first["charge"])
                if len(members) == 1 else 0.0,

            "mass":
                float(first["mass"])
                if len(members) == 1 else np.nan,

            "pT":
                float(first["pT"])
                if len(members) == 1 else np.nan,

            "eta":
                float(first["eta"])
                if len(members) == 1 else np.nan,

            "layer_id": layer,

            "trajectory_step":
                min(m["trajectory_step"] for m in members),

            "x": x,
            "y": y,
            "z": z,
            "r": R,
            "phi": phi,

            "pileup_z0": float(first["pileup_z0"]),

            "n_merged": len(members),
        })

pixel_merged = pd.DataFrame(merged_rows)

sct["n_merged"] = 1

out = pd.concat(
    [pixel_merged, sct],
    ignore_index=True,
    sort=False
)

out = out.sort_values(
    ["pileup_event", "layer_id", "trajectory_step"]
).reset_index(drop=True)

out.to_csv(FINAL, index=False)

print("\nSaved:", FINAL)

print("\n===== RESPONSE SUMMARY =====")

print("Initial hits:", len(df))
print("After pixelization:", len(pix))
print("After SCT response:", len(resp))
print("After pixel merging:", len(out))

print("\nPixel hits before merge:", len(pixel))
print("Pixel hits after merge:", len(pixel_merged))

print(
    "Removed by pixel merging:",
    len(pixel) - len(pixel_merged)
)

print(
    "Clusters with >1 hit:",
    int((pixel_merged["n_merged"] > 1).sum())
)

if len(pixel_merged):
    print(
        "Largest pixel cluster:",
        int(pixel_merged["n_merged"].max())
    )

print("\nFinal hits by layer:")
print(
    out["layer_id"]
       .value_counts()
       .sort_index()
)
