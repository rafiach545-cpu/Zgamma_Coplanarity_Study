from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import (
    paper_plane_finding_exact8,
    compute_T_tensor,
    _seed_pairs,
    _ensure_forward,
)

HITS = ROOT / "05_Tracking" / "converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"

EVENTS = [1900, 3485, 4790, 6899]
DS_FLOOR = 0.02

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

PAPER_LAYERS = [802,804,806,808,1302,1304,1306,1308]
GROWTH_LAYERS = [1304,1302,808,806,804,802]

def run_with_floor(xyz, layer_ids):
    outer_pairs = _seed_pairs(
        xyz, layer_ids == 1308,
        0.10, 20.0, max_pairs=50
    )
    second_pairs = _seed_pairs(
        xyz, layer_ids == 1306,
        0.10, 20.0, max_pairs=50
    )

    best = None

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

            cur_ds = fit["delta_s"]
            cur_dw = fit["delta_w"]

            if cur_ds >= 0.5 or cur_dw >= 10.0:
                continue

            accepted_idx = list(seed_idx)

            for lyr in GROWTH_LAYERS:
                for cand in np.where(layer_ids == lyr)[0]:

                    current_fit = compute_T_tensor(xyz[accepted_idx])

                    n1 = current_fit["n1"]
                    n2 = current_fit["n2"]
                    n3 = current_fit["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(xyz[accepted_idx], n3)
                    x = xyz[cand]

                    if (x @ n3) <= 0:
                        continue

                    if abs(x @ n1) >= 0.5:
                        continue

                    if abs(x @ n2) >= 10.0:
                        continue

                    trial_idx = accepted_idx + [cand]
                    trial_fit = compute_T_tensor(xyz[trial_idx])

                    t_ds = trial_fit["delta_s"]
                    t_dw = trial_fit["delta_w"]

                    effective_cur_ds = max(cur_ds, DS_FLOOR)

                    if t_ds > 3.0 * effective_cur_ds:
                        continue

                    if cur_dw > 0 and t_dw > 3.0 * cur_dw:
                        continue

                    accepted_idx = trial_idx
                    cur_ds = t_ds
                    cur_dw = t_dw

            final_fit = compute_T_tensor(xyz[accepted_idx])
            ds = final_fit["delta_s"]
            dw = final_fit["delta_w"]

            counts = {
                lyr: int(np.sum(layer_ids[accepted_idx] == lyr))
                for lyr in PAPER_LAYERS
            }

            found = (
                ds < 0.1
                and dw < 10.0
                and all(counts[lyr] >= 1 for lyr in PAPER_LAYERS)
                and sum(counts[lyr] >= 2 for lyr in PAPER_LAYERS) >= 7
            )

            result = {
                "found": found,
                "delta_s": ds,
                "delta_w": dw,
                "accepted_hits": len(accepted_idx),
            }

            if best is None or ds < best["delta_s"]:
                best = result

            if found:
                return result

    return best if best is not None else {
        "found": False,
        "delta_s": np.nan,
        "delta_w": np.nan,
        "accepted_hits": 0,
    }

df = pd.read_csv(HITS)

rows = []

for event in EVENTS:
    g = df[df["event"] == event].copy()
    g["logical_layer"] = g["layer_id"].astype(int).map(MAP)

    xyz = g[["x","y","z"]].to_numpy(dtype=float)
    layers = g["logical_layer"].to_numpy()

    orig = paper_plane_finding_exact8(xyz, layers)
    floor = run_with_floor(xyz, layers)

    rows.append({
        "event": event,
        "orig_found": bool(orig.get("found", False)),
        "orig_ds": orig.get("delta_s", np.nan),
        "floor_found": bool(floor.get("found", False)),
        "floor_ds": floor.get("delta_s", np.nan),
        "floor_dw": floor.get("delta_w", np.nan),
        "floor_accepted_hits": floor.get("accepted_hits", 0),
    })

out = pd.DataFrame(rows)

print("\n===== 4 FLOOR-SENSITIVE BACKGROUND EVENTS =====")
print(out.to_string(index=False))

print("\nClassification changes:")
print(int((out["orig_found"] != out["floor_found"]).sum()))

OUTPUT = ROOT / "06_Coplanarity" / "floor_sensitive_background_comparison.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
