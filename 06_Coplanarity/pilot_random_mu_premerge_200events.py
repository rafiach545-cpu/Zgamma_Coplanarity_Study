import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"

OUT = "06_Coplanarity/pilot_random_mu_premerge_200events.csv"

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

KNOWN_PASS = [405, 5833, 8053]
MUS = [5, 10, 20]

SEED = 12345

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

all_hard_events = np.array(sorted(hard["event"].unique()))
all_pu_events = np.array(sorted(pu["pileup_event"].unique()))

rng = np.random.default_rng(SEED)

others = all_hard_events[
    ~np.isin(all_hard_events, KNOWN_PASS)
]

chosen_random = rng.choice(
    others,
    size=197,
    replace=False
)

TEST_EVENTS = np.concatenate([
    np.array(KNOWN_PASS),
    chosen_random
])

keep = [
    "layer_id",
    "trajectory_step",
    "x", "y", "z", "r", "phi"
]

def run_plane(df):
    xyz = df[["x","y","z"]].to_numpy(float)

    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )

    return paper_plane_finding_exact8(
        xyz,
        layers
    )

rows = []

for n, event in enumerate(TEST_EVENTS, start=1):

    h = hard[
        hard["event"] == event
    ].copy()

    rh = run_plane(h)

    hard_found = bool(
        rh.get("found", False)
    )

    base_row = {
        "event": int(event),
        "mu": 0,
        "hard_found": hard_found,
        "pu_found": hard_found,
        "transition": f"{int(hard_found)}->{int(hard_found)}",
        "hard_ds": rh.get("delta_s", np.nan),
        "pu_ds": rh.get("delta_s", np.nan),
        "hard_dw": rh.get("delta_w", np.nan),
        "pu_dw": rh.get("delta_w", np.nan),
        "hard_hits": len(h),
        "combined_hits": len(h),
        "pu_events": ""
    }

    rows.append(base_row)

    for mu in MUS:

        selected_pu = rng.choice(
            all_pu_events,
            size=mu,
            replace=False
        )

        p = pu[
            pu["pileup_event"].isin(selected_pu)
        ][keep].copy()

        combo = pd.concat(
            [h[keep], p],
            ignore_index=True
        )

        rp = run_plane(combo)

        pu_found = bool(
            rp.get("found", False)
        )

        rows.append({
            "event": int(event),
            "mu": int(mu),
            "hard_found": hard_found,
            "pu_found": pu_found,
            "transition":
                f"{int(hard_found)}->{int(pu_found)}",
            "hard_ds": rh.get("delta_s", np.nan),
            "pu_ds": rp.get("delta_s", np.nan),
            "hard_dw": rh.get("delta_w", np.nan),
            "pu_dw": rp.get("delta_w", np.nan),
            "hard_hits": len(h),
            "combined_hits": len(combo),
            "pu_events":
                ",".join(map(str, selected_pu))
        })

    if n % 10 == 0:
        print(f"Processed {n}/200 hard events")

out = pd.DataFrame(rows)

print("\n===== TRANSITIONS BY MU =====")

for mu in [0, 5, 10, 20]:

    print(f"\n--- mu={mu} ---")

    x = out[out["mu"] == mu]

    print(
        x["transition"]
        .value_counts()
        .sort_index()
    )

    print(
        "Found:",
        int(x["pu_found"].sum()),
        "/",
        len(x)
    )

print("\n===== KNOWN PASS EVENTS =====")

print(
    out[
        out["event"].isin(KNOWN_PASS)
    ][
        [
            "event",
            "mu",
            "hard_found",
            "pu_found",
            "transition",
            "hard_ds",
            "pu_ds",
            "hard_dw",
            "pu_dw",
            "combined_hits"
        ]
    ].to_string(index=False)
)

print("\n===== NEW PU-INDUCED POSITIVES =====")

fake = out[
    (out["mu"] > 0) &
    (~out["hard_found"]) &
    (out["pu_found"])
]

print("Count:", len(fake))

if len(fake):
    print(
        fake[
            [
                "event",
                "mu",
                "pu_ds",
                "pu_dw",
                "combined_hits",
                "pu_events"
            ]
        ].to_string(index=False)
    )

out.to_csv(
    OUT,
    index=False
)

print("\nSaved:", OUT)
