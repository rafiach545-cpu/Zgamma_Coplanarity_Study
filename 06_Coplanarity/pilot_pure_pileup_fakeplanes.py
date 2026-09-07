import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

PU = "05_Tracking/minbias_1000_hits_detector_response.csv"
OUT = "06_Coplanarity/pilot_pure_pileup_fakeplanes.csv"

MUS = [5, 10, 20, 50]
N_PSEUDO = 200
SEED = 24680

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

KEEP = ["layer_id", "x", "y", "z"]

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

for mu in MUS:

    print(f"\n===== mu={mu} =====")

    for i in range(N_PSEUDO):

        selected = rng.choice(
            all_events,
            size=mu,
            replace=False
        )

        combo = pu[
            pu["pileup_event"].isin(selected)
        ][KEEP].copy()

        r = run_plane(combo)

        rows.append({
            "pseudo_event": i,
            "mu": mu,
            "found": bool(r.get("found", False)),
            "delta_s": r.get("delta_s", np.nan),
            "delta_w": r.get("delta_w", np.nan),
            "hits": len(combo),
            "pu_events": ",".join(map(str, selected))
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{N_PSEUDO}")

out = pd.DataFrame(rows)

print("\n===== PURE PU RESULTS =====")

for mu in MUS:
    x = out[out["mu"] == mu]

    print(
        f"mu={mu}: "
        f"{int(x['found'].sum())}/{len(x)} "
        f"fake planes"
    )

out.to_csv(OUT, index=False)
print("\nSaved:", OUT)
