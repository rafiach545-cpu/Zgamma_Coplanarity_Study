import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"

OUT   = "06_Coplanarity/full_mu20_premerge.csv"
CKPT  = "06_Coplanarity/full_mu20_premerge_checkpoint.csv"
STATE = "06_Coplanarity/full_mu20_premerge_state.txt"

MU = 20
SEED = 12345

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

KEEP = [
    "layer_id",
    "trajectory_step",
    "x","y","z","r","phi"
]

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

events = np.array(sorted(hard["event"].unique()))
pu_events_all = np.array(sorted(pu["pileup_event"].unique()))

def run_plane(df):
    xyz = df[["x","y","z"]].to_numpy(float)
    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )
    return paper_plane_finding_exact8(xyz, layers)

# Deterministic PU choices for every hard event
rng = np.random.default_rng(SEED)

selections = {}
for ev in events:
    selections[int(ev)] = rng.choice(
        pu_events_all,
        size=MU,
        replace=False
    )

start = 0

if os.path.exists(STATE):
    with open(STATE) as f:
        txt = f.read().strip()
        if txt:
            start = int(txt)

print("Hard events:", len(events))
print("PU pool events:", len(pu_events_all))
print("mu:", MU)
print("Resuming from event index:", start)

write_header = not os.path.exists(CKPT)

for i in range(start, len(events)):

    event = int(events[i])

    h = hard[hard["event"] == event].copy()

    rh = run_plane(h)
    hard_found = bool(rh.get("found", False))

    selected = selections[event]

    p = pu[
        pu["pileup_event"].isin(selected)
    ][KEEP].copy()

    combo = pd.concat(
        [h[KEEP], p],
        ignore_index=True
    )

    rp = run_plane(combo)
    pu_found = bool(rp.get("found", False))

    row = pd.DataFrame([{
        "event": event,
        "mu": MU,

        "hard_found": hard_found,
        "pu_found": pu_found,

        "transition":
            f"{int(hard_found)}->{int(pu_found)}",

        "hard_ds":
            rh.get("delta_s", np.nan),

        "pu_ds":
            rp.get("delta_s", np.nan),

        "hard_dw":
            rh.get("delta_w", np.nan),

        "pu_dw":
            rp.get("delta_w", np.nan),

        "hard_hits": len(h),
        "combined_hits": len(combo),

        "pu_events":
            ",".join(map(str, selected))
    }])

    row.to_csv(
        CKPT,
        mode="a",
        header=write_header,
        index=False
    )

    write_header = False

    with open(STATE, "w") as f:
        f.write(str(i + 1))

    if (i + 1) % 100 == 0:
        print(f"Checkpoint saved: {i+1}/{len(events)}")

# Finalize
df = pd.read_csv(CKPT)
df.to_csv(OUT, index=False)

print("\n===== FINAL mu=20 =====")
print(
    df["transition"]
    .value_counts()
    .sort_index()
)

print(
    "\nPass:",
    int(df["pu_found"].sum()),
    "/",
    len(df)
)

print(
    "New 0->1:",
    int((df["transition"] == "0->1").sum())
)

print(
    "Lost 1->0:",
    int((df["transition"] == "1->0").sum())
)

print("\nSaved:", OUT)
