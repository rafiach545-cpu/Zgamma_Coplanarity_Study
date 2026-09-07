import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")

from coplanarity import (
    compute_T_tensor,
    _seed_pairs,
    _ensure_forward,
)

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
df = pd.read_csv(INPUT)

PAPER_LAYERS = [802,804,806,808,1302,1304,1306,1308]
OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

vals = []

for event, ev in df.groupby("event"):

    xyz = ev[["x","y","z"]].to_numpy(float)
    layer_ids = np.array([MAP[int(x)] for x in ev["layer_id"]], dtype=int)

    outer_pairs = _seed_pairs(xyz, layer_ids == OUTER, 0.10, 20.0, max_pairs=50)
    second_pairs = _seed_pairs(xyz, layer_ids == SECOND, 0.10, 20.0, max_pairs=50)

    if not outer_pairs or not second_pairs:
        continue

    best_final = None

    for _, oi, oj in outer_pairs:
        for _, si, sj in second_pairs:

            seed_idx = [oi,oj,si,sj]
            seed_xyz = xyz[seed_idx]

            fit = compute_T_tensor(seed_xyz)

            if fit["n3"] is None:
                continue

            n3 = _ensure_forward(seed_xyz, fit["n3"])

            if not np.all((seed_xyz @ n3) > 0):
                continue

            cur_ds = float(fit["delta_s"])
            cur_dw = float(fit["delta_w"])

            if cur_ds >= 0.5 or cur_dw >= 10.0:
                continue

            accepted_idx = list(seed_idx)

            for lyr in GROWTH:
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

                    trial = accepted_idx + [cand]
                    tf = compute_T_tensor(xyz[trial])

                    tds = float(tf["delta_s"])
                    tdw = float(tf["delta_w"])

                    if cur_ds > 0 and tds > 3.0 * cur_ds:
                        continue
                    if cur_dw > 0 and tdw > 3.0 * cur_dw:
                        continue

                    accepted_idx = trial
                    cur_ds = tds
                    cur_dw = tdw

            ff = compute_T_tensor(xyz[accepted_idx])

            ds = float(ff["delta_s"])
            dw = float(ff["delta_w"])

            if best_final is None or ds < best_final[0]:
                best_final = (ds, dw, len(accepted_idx))

    if best_final is not None:
        vals.append({
            "event": event,
            "final_ds_mm": best_final[0],
            "final_dw_mm": best_final[1],
            "n_hits": best_final[2]
        })

out = pd.DataFrame(vals)

print("Events with seed passing:", len(out))

print("\nFinal delta_s distribution (mm):")
print(out["final_ds_mm"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))

print("\nCounts relative to final cut 0.1 mm:")
for cut in [0.1,0.15,0.2,0.3,0.5,1.0]:
    print(f"ds < {cut:.2f} mm:", int((out["final_ds_mm"] < cut).sum()))

print("\nSmallest 20 final ds values:")
print(out.sort_values("final_ds_mm").head(20).to_string(index=False))

out.to_csv(
    "06_Coplanarity/final_ds_distribution_full_response.csv",
    index=False
)

print("\nSaved: 06_Coplanarity/final_ds_distribution_full_response.csv")
