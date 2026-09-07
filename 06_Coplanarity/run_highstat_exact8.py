from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import paper_plane_finding_exact8

hits_path = ROOT / "05_Tracking" / "converted_photon_hits_highstat_curved.csv"
out_path = ROOT / "06_Coplanarity" / "conversion_coplanarity_highstat_exact8.csv"

hits = pd.read_csv(hits_path)

results = []

for event, ev in hits.groupby("event"):

    xyz = ev[["x", "y", "z"]].to_numpy()

    layers = (
        ev["volume_id"].astype(int) * 100
        + ev["layer_id"].astype(int)
    ).to_numpy()

    result = paper_plane_finding_exact8(
        xyz,
        layers
    )

    results.append({
        "event": event,
        "found": bool(result.get("found", False)),
        "delta_s": result.get("delta_s", None),
        "delta_w": result.get("delta_w", None),
        "n_hits": len(ev),
        "reason": result.get("reason", "")
    })

df = pd.DataFrame(results)

df.to_csv(out_path, index=False)

print()
print("Saved:", out_path)
print("Events:", len(df))
print("Found:", int(df["found"].sum()))
print("Acceptance:", df["found"].mean())

print("\nFailure reasons:")
print(df.loc[~df["found"], "reason"].value_counts())
