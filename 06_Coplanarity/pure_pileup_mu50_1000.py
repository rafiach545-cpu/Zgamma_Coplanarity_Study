import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

PU = "05_Tracking/minbias_1000_hits_detector_response.csv"
OUT = "06_Coplanarity/pure_pileup_mu50_1000.csv"

MU = 50
N_PSEUDO = 1000
SEED = 97531

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

pu = pd.read_csv(PU)
all_events = np.array(sorted(pu["pileup_event"].unique()))

rng = np.random.default_rng(SEED)

def run_plane(df):
    xyz = df[["x","y","z"]].to_numpy(float)
    layers = np.array(
        [MAP[int(x)] for x in df["layer_id"]],
        dtype=int
    )
    return paper_plane_finding_exact8(xyz, layers)

rows = []

for i in range(N_PSEUDO):

    selected = rng.choice(
        all_events,
        size=MU,
        replace=False
    )

    combo = pu[
        pu["pileup_event"].isin(selected)
    ]

    r = run_plane(combo)

    rows.append({
        "pseudo_event": i,
        "mu": MU,
        "found": bool(r.get("found", False)),
        "delta_s": r.get("delta_s", np.nan),
        "delta_w": r.get("delta_w", np.nan),
        "hits": len(combo),
        "pu_events": ",".join(map(str, selected))
    })

    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{N_PSEUDO}")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("\n===== PURE PU mu=50 =====")
print("Fake planes:", int(out["found"].sum()), "/", len(out))
print("Saved:", OUT)
