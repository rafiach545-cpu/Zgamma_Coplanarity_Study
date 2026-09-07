import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_10_hits_detector_response.csv"

OUT = "06_Coplanarity/pilot_mu5_premerge_200events.csv"

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

PU_EVENTS = [1,2,3,4,5]
KNOWN_PASS = [405,5833,8053]

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

all_events = np.array(sorted(hard["event"].unique()))

rng = np.random.default_rng(12345)

others = all_events[
    ~np.isin(all_events, KNOWN_PASS)
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

p = pu[
    pu["pileup_event"].isin(PU_EVENTS)
].copy()

keep = [
    "layer_id",
    "trajectory_step",
    "x","y","z","r","phi"
]

p = p[keep].copy()

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

    combo = pd.concat(
        [h[keep], p],
        ignore_index=True
    )

    rp = run_plane(combo)

    hard_found = bool(
        rh.get("found", False)
    )

    pu_found = bool(
        rp.get("found", False)
    )

    transition = (
        f"{int(hard_found)}->{int(pu_found)}"
    )

    rows.append({
        "event": event,
        "hard_found": hard_found,
        "plus5pu_found": pu_found,
        "transition": transition,
        "hard_ds": rh.get("delta_s", np.nan),
        "plus5pu_ds": rp.get("delta_s", np.nan),
        "hard_dw": rh.get("delta_w", np.nan),
        "plus5pu_dw": rp.get("delta_w", np.nan),
        "hard_hits": len(h),
        "combined_hits": len(combo),
    })

    if n % 25 == 0:
        print(f"Processed {n}/200")

out = pd.DataFrame(rows)

print("\n===== TRANSITIONS =====")
print(
    out["transition"]
    .value_counts()
    .sort_index()
)

print("\nDefinitions:")
print("0->0 : remains failing")
print("0->1 : NEW pile-up-induced positive")
print("1->0 : genuine plane lost")
print("1->1 : genuine plane survives")

print("\n===== KNOWN PASS EVENTS =====")
print(
    out[
        out["event"].isin(KNOWN_PASS)
    ][
        [
            "event",
            "hard_found",
            "plus5pu_found",
            "hard_ds",
            "plus5pu_ds"
        ]
    ].to_string(index=False)
)

new_fake = out[
    (~out["hard_found"]) &
    (out["plus5pu_found"])
]

print("\nNew PU-induced positives:", len(new_fake))

if len(new_fake):
    print(
        new_fake[
            [
                "event",
                "plus5pu_ds",
                "plus5pu_dw"
            ]
        ].to_string(index=False)
    )

out.to_csv(
    OUT,
    index=False
)

print("\nSaved:", OUT)
