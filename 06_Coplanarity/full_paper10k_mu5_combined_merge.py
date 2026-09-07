import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_eta18_detector_response_beamspot.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"

MU = 5
SEED = 12345
NTEST = 7388

PITCH_S = {1:0.050, 2:0.050, 3:0.050, 4:0.050}
PITCH_Z = {1:0.250, 2:0.400, 3:0.400, 4:0.400}

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

hard_events = sorted(hard["event"].unique())[:NTEST]
pu_events = np.array(sorted(pu["pileup_event"].unique()))

rng = np.random.default_rng(SEED)

def merge_combined_pixel_hits(df):

    pixel = df[df["layer_id"].isin([1,2,3,4])].copy()
    sct = df[df["layer_id"].isin([5,6,7,8])].copy()

    merged = []

    # One combined detector event: group only by layer.
    for layer, g in pixel.groupby("layer_id", sort=False):

        layer = int(layer)
        R = float(g["r"].iloc[0])

        ps = PITCH_S[layer]
        pz = PITCH_Z[layer]

        g = g.copy()
        g["pix_s"] = np.rint((R * g["phi"]) / ps).astype(int)
        g["pix_z"] = np.rint(g["z"] / pz).astype(int)

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

            x = np.mean([m["x"] for m in members])
            y = np.mean([m["y"] for m in members])
            z = np.mean([m["z"] for m in members])

            merged.append({
                "layer_id": layer,
                "x": x,
                "y": y,
                "z": z,
                "r": R,
                "phi": np.arctan2(y, x),
                "n_merged": len(members),
            })

    pm = pd.DataFrame(merged)

    sct_out = sct[["layer_id","x","y","z","r","phi"]].copy()
    sct_out["n_merged"] = 1

    return pd.concat([pm, sct_out], ignore_index=True)


def run_plane(df, vertex_z):

    xyz = df[["x","y","z"]].to_numpy(float)

    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )

    return paper_plane_finding_exact8(
        xyz,
        layers,
        vertex_z=vertex_z,
    )


rows = []

for n, ev in enumerate(hard_events, 1):

    h = hard[hard["event"] == ev].copy()

    z0_hard = float(h["beamspot_z0"].iloc[0])

    r_hard = run_plane(h, z0_hard)
    hard_found = bool(r_hard.get("found", False))

    selected = rng.choice(
        pu_events,
        size=MU,
        replace=False
    )

    p = pu[pu["pileup_event"].isin(selected)].copy()

    combo = pd.concat(
        [
            h[["layer_id","x","y","z","r","phi"]],
            p[["layer_id","x","y","z","r","phi"]],
        ],
        ignore_index=True
    )

    merged = merge_combined_pixel_hits(combo)

    r_combo = run_plane(merged, z0_hard)
    combo_found = bool(r_combo.get("found", False))

    rows.append({
        "event": ev,
        "hard_found": hard_found,
        "combined_found": combo_found,
        "transition": f"{int(hard_found)}->{int(combo_found)}",
        "hard_hits": len(h),
        "combined_premerge_hits": len(combo),
        "combined_postmerge_hits": len(merged),
        "combined_ds": r_combo.get("delta_s", np.nan),
        "combined_dw": r_combo.get("delta_w", np.nan),
    })

    if n % 20 == 0:
        print(f"Processed {n}/{NTEST}")

out = pd.DataFrame(rows)

print("\n=== mu=5 COMBINED-MERGE SMOKE ===")
print(out["transition"].value_counts().sort_index())
print()
print("Pass:", int(out["combined_found"].sum()), "/", len(out))
print("New 0->1:", int((out["transition"]=="0->1").sum()))
print("Lost 1->0:", int((out["transition"]=="1->0").sum()))
print()
print("Mean hits before combined merge:", out["combined_premerge_hits"].mean())
print("Mean hits after combined merge :", out["combined_postmerge_hits"].mean())

out.to_csv(
    "06_Coplanarity/paper10k_mu5_combined_merge_full.csv",
    index=False
)
