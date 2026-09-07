import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/quirk_hits_methodB_pixel_only_200gev.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"

MU = 50
SEED = 12345

PITCH_S = {1:0.050, 2:0.050, 3:0.050, 4:0.050}
PITCH_Z = {1:0.250, 2:0.400, 3:0.400, 4:0.400}

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

hard_events = sorted(hard["event"].unique())
pu_events = np.array(sorted(pu["pileup_event"].unique()))

rng = np.random.default_rng(SEED)

def merge_combined_pixel_hits(df):
    pixel = df[df["layer_id"].isin([1,2,3,4])].copy()
    sct = df[df["layer_id"].isin([5,6,7,8])].copy()

    merged = []

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
        vertex_z=vertex_z
    )

rows = []

for ev in hard_events:
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

    combo = pd.concat([
        h[["layer_id","x","y","z","r","phi"]],
        p[["layer_id","x","y","z","r","phi"]]
    ], ignore_index=True)

    merged = merge_combined_pixel_hits(combo)

    r_combo = run_plane(merged, z0_hard)
    combo_found = bool(r_combo.get("found", False))

    accepted_hits = len(r_combo.get("inlier_idx", [])) if combo_found else 0
    accepted_fraction = accepted_hits / len(merged) if len(merged) else 0.0
    layer_counts = r_combo.get("layer_counts", {}) if combo_found else {}
    delta_s = r_combo.get("delta_s", float("nan")) if combo_found else float("nan")

    rows.append({
        "event": int(ev),
        "hard_found": hard_found,
        "combined_found": combo_found,
        "transition": f"{int(hard_found)}->{int(combo_found)}",
        "premerge_hits": len(combo),
        "postmerge_hits": len(merged),
        "accepted_hits": accepted_hits,
        "accepted_fraction": accepted_fraction,
        "delta_s": delta_s,
        "layer_counts": str(layer_counts),
    })

out = pd.DataFrame(rows)

print("\n=== mu=50 QUIRK PIXEL-ONLY + PU + MERGE ===")
print(out["transition"].value_counts().sort_index())

print("\nPass:", int(out["combined_found"].sum()), "/", len(out))
print("New 0->1:", int((out["transition"]=="0->1").sum()))
print("Lost 1->0:", int((out["transition"]=="1->0").sum()))

print("\nMean hits premerge :", out["premerge_hits"].mean())
print("Mean hits postmerge:", out["postmerge_hits"].mean())

out.to_csv(
    "06_Coplanarity/quirk_mu50_acceptance_diagnostic.csv",
    index=False
)

print("\nSaved: 06_Coplanarity/quirk_pixelonly_merged_mu10_200gev.csv")
