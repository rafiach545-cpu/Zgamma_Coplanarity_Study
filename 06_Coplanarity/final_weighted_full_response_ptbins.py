import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

HITS = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
META = "04_Conversion/converted_pairs_weighted_barrel_eta18.csv"

hits = pd.read_csv(HITS)
meta = pd.read_csv(META)

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

# one row per conversion event
meta_event = (
    meta.groupby("event", as_index=False)
        .first()
)

rows = []

for n, row in meta_event.iterrows():

    event = row["event"]
    ev = hits[hits["event"] == event]

    found = False

    if len(ev) > 0:
        xyz = ev[["x","y","z"]].to_numpy(float)

        layers = np.array(
            [MAP[int(x)] for x in ev["layer_id"]],
            dtype=int
        )

        r = paper_plane_finding_exact8(
            xyz,
            layers
        )

        found = bool(r.get("found", False))

    rows.append({
        "event": event,
        "photon_pT": float(row["photon_pT"]),
        "weight": float(row["weight"]),
        "found": found
    })

    if (n+1) % 500 == 0:
        print(f"Processed {n+1}/{len(meta_event)}")

out = pd.DataFrame(rows)

out["pT_bin"] = pd.cut(
    out["photon_pT"],
    bins=[-np.inf, 5, 20, np.inf],
    labels=["<5", "5-20", ">20"],
    right=False
)

print("\n===== FULL RESPONSE WEIGHTED RESULT =====")

total_weight = out["weight"].sum()
pass_weight = out.loc[out["found"], "weight"].sum()

print("Events:", len(out))
print("Found:", int(out["found"].sum()))
print("Total effective weight:", total_weight)
print("Passed effective weight:", pass_weight)
print("Weighted acceptance:", pass_weight / total_weight)

summary = (
    out.groupby("pT_bin", observed=True)
       .agg(
           n_total=("event","count"),
           n_pass=("found","sum"),
           effective_total=("weight","sum"),
           effective_pass=("weight",
               lambda x: x[out.loc[x.index,"found"]].sum())
       )
       .reset_index()
)

summary["weighted_acceptance"] = (
    summary["effective_pass"] /
    summary["effective_total"]
)

print("\n===== pT BINS =====")
print(summary.to_string(index=False))

out.to_csv(
    "06_Coplanarity/full_response_weighted_events.csv",
    index=False
)

summary.to_csv(
    "06_Coplanarity/full_response_weighted_ptbins.csv",
    index=False
)

print("\nSaved:")
print("06_Coplanarity/full_response_weighted_events.csv")
print("06_Coplanarity/full_response_weighted_ptbins.csv")
