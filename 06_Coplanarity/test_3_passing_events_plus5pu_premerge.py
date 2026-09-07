import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_10_hits_detector_response.csv"

PASS_EVENTS = [405, 5833, 8053]
PU_EVENTS = [1,2,3,4,5]

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

rows = []

def run_plane(df):
    xyz = df[["x","y","z"]].to_numpy(float)
    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )
    return paper_plane_finding_exact8(xyz, layers)

for event in PASS_EVENTS:

    h = hard[hard["event"] == event].copy()
    p = pu[pu["pileup_event"].isin(PU_EVENTS)].copy()

    keep = [
        "layer_id",
        "trajectory_step",
        "x","y","z","r","phi"
    ]

    combo = pd.concat(
        [h[keep], p[keep]],
        ignore_index=True
    )

    rh = run_plane(h)
    rp = run_plane(combo)

    print("\n==============================")
    print("Event:", event)
    print("Hard hits:", len(h))
    print("+5PU hits:", len(combo))
    print("Hard alone:", rh.get("found", False))
    print("Pre-merge +5PU:", rp.get("found", False))

    if rp.get("found", False):
        print("ds:", rp.get("delta_s"))
        print("dw:", rp.get("delta_w"))
    else:
        print("reason:", rp.get("reason"))

    rows.append({
        "event": event,
        "hard_found": rh.get("found", False),
        "plus5pu_premerge_found": rp.get("found", False),
        "hard_hits": len(h),
        "combined_hits": len(combo),
    })

out = pd.DataFrame(rows)

print("\n===== SUMMARY =====")
print(out.to_string(index=False))

out.to_csv(
    "06_Coplanarity/three_passing_events_plus5pu_premerge.csv",
    index=False
)

print("\nSaved:")
print("06_Coplanarity/three_passing_events_plus5pu_premerge.csv")
