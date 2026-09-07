from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import compute_T_tensor, _seed_pairs, _ensure_forward

HITS = ROOT / "05_Tracking" / "converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"

MAP = {
    1: 802, 2: 804, 3: 806, 4: 808,
    5: 1302, 6: 1304, 7: 1306, 8: 1308,
}

df = pd.read_csv(HITS)

rows = []

for event_id, g in df.groupby("event"):

    q = g.copy()
    q["logical_layer"] = q["layer_id"].astype(int).map(MAP)

    xyz = q[["x","y","z"]].to_numpy(dtype=float)
    layers = q["logical_layer"].to_numpy()

    outer_pairs = _seed_pairs(
        xyz, layers == 1308,
        0.10, 20.0, max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz, layers == 1306,
        0.10, 20.0, max_pairs=50
    )

    seed_ds_values = []

    for _, oi, oj in outer_pairs:
        for _, si, sj in second_pairs:

            seed_idx = [oi, oj, si, sj]
            seed_xyz = xyz[seed_idx]

            fit = compute_T_tensor(seed_xyz)

            if fit["n3"] is None:
                continue

            n3 = _ensure_forward(seed_xyz, fit["n3"])

            if not np.all((seed_xyz @ n3) > 0):
                continue

            ds = fit["delta_s"]
            dw = fit["delta_w"]

            if ds >= 0.5 or dw >= 10.0:
                continue

            seed_ds_values.append(ds)

    if seed_ds_values:
        rows.append({
            "event": int(event_id),
            "n_valid_seeds": len(seed_ds_values),
            "min_seed_ds": min(seed_ds_values),
            "median_seed_ds": float(np.median(seed_ds_values)),
            "max_seed_ds": max(seed_ds_values),
        })

out = pd.DataFrame(rows)

print("\n===== BACKGROUND SEED DELTA_S =====")
print("Events with valid seed:", len(out))

if len(out):
    print("\nMinimum seed ds overall:")
    print(out["min_seed_ds"].min())

    print("\nMedian of event minimum seed ds:")
    print(out["min_seed_ds"].median())

    print("\nEvents with min seed ds < 0.02 mm:")
    n_floor = int((out["min_seed_ds"] < 0.02).sum())
    print(n_floor, "/", len(out))

    print("\nEvents with min seed ds < 0.01 mm:")
    print(int((out["min_seed_ds"] < 0.01).sum()), "/", len(out))

    print("\nQuantiles of min seed ds:")
    print(
        out["min_seed_ds"].quantile(
            [0.0,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.0]
        )
    )

OUTPUT = ROOT / "06_Coplanarity" / "background_seed_ds_diagnostic.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
