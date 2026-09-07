import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"
SEL  = "06_Coplanarity/pilot_random_mu_premerge_200events.csv"

OUT = "06_Coplanarity/pilot_random_mu_prepost_200events.csv"

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

PITCH_S = {1:0.050, 2:0.050, 3:0.050, 4:0.050}
PITCH_Z = {1:0.250, 2:0.400, 3:0.400, 4:0.400}

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)
sel = pd.read_csv(SEL)

def run_plane(df):
    xyz = df[["x","y","z"]].to_numpy(float)
    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )
    return paper_plane_finding_exact8(xyz, layers)

def joint_pixel_merge(df):

    pixel = df[df["layer_id"].isin([1,2,3,4])].copy()
    sct   = df[df["layer_id"].isin([5,6,7,8])].copy()

    merged = []

    for layer, g in pixel.groupby("layer_id", sort=False):

        layer = int(layer)
        R = float(g["r"].iloc[0])

        ps = PITCH_S[layer]
        pz = PITCH_Z[layer]

        g = g.copy().reset_index(drop=True)

        g["pix_s"] = np.rint(
            (R * g["phi"]) / ps
        ).astype(int)

        g["pix_z"] = np.rint(
            g["z"] / pz
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

            merged.append({
                "layer_id": layer,
                "x": x,
                "y": y,
                "z": z,
                "r": R,
                "phi": np.arctan2(y, x),
                "n_merged": len(members)
            })

    pixel_out = pd.DataFrame(merged)

    sct_out = sct[
        ["layer_id","x","y","z","r","phi"]
    ].copy()

    sct_out["n_merged"] = 1

    return pd.concat(
        [pixel_out, sct_out],
        ignore_index=True,
        sort=False
    )

rows = []

settings = sel[
    ["event","mu","pu_events"]
].drop_duplicates()

N = len(settings)

for n, r in enumerate(settings.itertuples(index=False), start=1):

    event = int(r.event)
    mu = int(r.mu)

    h = hard[hard["event"] == event].copy()

    if mu == 0:
        p = pu.iloc[0:0].copy()
    else:
        ids = [
            int(x)
            for x in str(r.pu_events).split(",")
            if x.strip()
        ]

        p = pu[
            pu["pileup_event"].isin(ids)
        ].copy()

    keep = [
        "layer_id","x","y","z","r","phi"
    ]

    combo = pd.concat(
        [h[keep], p[keep]],
        ignore_index=True
    )

    pre = run_plane(combo)

    merged = joint_pixel_merge(combo)

    post = run_plane(merged)

    rows.append({
        "event": event,
        "mu": mu,

        "pre_found":
            bool(pre.get("found", False)),

        "post_found":
            bool(post.get("found", False)),

        "pre_ds":
            pre.get("delta_s", np.nan),

        "post_ds":
            post.get("delta_s", np.nan),

        "pre_dw":
            pre.get("delta_w", np.nan),

        "post_dw":
            post.get("delta_w", np.nan),

        "hits_pre":
            len(combo),

        "hits_post":
            len(merged),

        "merged_away":
            len(combo) - len(merged),
    })

    if n % 50 == 0:
        print(f"Processed {n}/{N}")

out = pd.DataFrame(rows)

print("\n===== PRE / POST RESULTS =====")

for mu in [0,5,10,20]:

    x = out[out["mu"] == mu]

    print(f"\n--- mu={mu} ---")
    print(
        "PRE :",
        int(x["pre_found"].sum()),
        "/",
        len(x)
    )
    print(
        "POST:",
        int(x["post_found"].sum()),
        "/",
        len(x)
    )
    print(
        "Pixel hits merged away:",
        int(x["merged_away"].sum())
    )

print("\n===== PRE -> POST TRANSITIONS =====")

out["merge_transition"] = (
    out["pre_found"].astype(int).astype(str)
    + "->" +
    out["post_found"].astype(int).astype(str)
)

for mu in [0,5,10,20]:
    print(f"\nmu={mu}")
    print(
        out[out["mu"] == mu][
            "merge_transition"
        ].value_counts().sort_index()
    )

print("\n===== EVENT 405 =====")
print(
    out[out["event"] == 405][
        [
            "event","mu",
            "pre_found","post_found",
            "pre_ds","post_ds",
            "hits_pre","hits_post",
            "merged_away"
        ]
    ].to_string(index=False)
)

out.to_csv(OUT, index=False)
print("\nSaved:", OUT)
