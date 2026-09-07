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

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

records = []

for event, ev in df.groupby("event"):

    xyz = ev[["x","y","z"]].to_numpy(float)
    layers = np.array(
        [MAP[int(x)] for x in ev["layer_id"]],
        dtype=int
    )

    p8 = _seed_pairs(
        xyz, layers == OUTER,
        0.10, 20.0, max_pairs=50
    )

    p7 = _seed_pairs(
        xyz, layers == SECOND,
        0.10, 20.0, max_pairs=50
    )

    if not p8 or not p7:
        continue

    for _, oi, oj in p8:
        for _, si, sj in p7:

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

                candidates = np.where(layers == lyr)[0]

                for cand in candidates:

                    cf = compute_T_tensor(
                        xyz[accepted_idx]
                    )

                    n1 = cf["n1"]
                    n2 = cf["n2"]
                    n3 = cf["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(
                        xyz[accepted_idx],
                        n3
                    )

                    x = xyz[cand]

                    n1dist = abs(float(x @ n1))

                    if lyr in [1304,1302]:
                        records.append({
                            "event": event,
                            "layer": lyr,
                            "n1_distance_mm": n1dist,
                            "passes_0p5": n1dist < 0.5
                        })

                    if (x @ n3) <= 0:
                        continue

                    if n1dist >= 0.5:
                        continue

                    if abs(x @ n2) >= 10.0:
                        continue

                    trial = accepted_idx + [cand]
                    tf = compute_T_tensor(xyz[trial])

                    tds = float(tf["delta_s"])
                    tdw = float(tf["delta_w"])

                    if cur_ds > 0 and tds > 3.0*cur_ds:
                        continue

                    if cur_dw > 0 and tdw > 3.0*cur_dw:
                        continue

                    accepted_idx = trial
                    cur_ds = tds
                    cur_dw = tdw

out = pd.DataFrame(records)

print("===== |x.n1| DISTRIBUTION =====")

for lyr in [1304,1302]:

    x = out[out["layer"] == lyr]["n1_distance_mm"]

    print()
    print("Layer", lyr)
    print("count:", len(x))
    print("median:", x.median())
    print("75%:", x.quantile(0.75))
    print("90%:", x.quantile(0.90))
    print("95%:", x.quantile(0.95))
    print("99%:", x.quantile(0.99))

    for cut in [0.5,0.6,0.8,1.0,1.5,2.0]:
        print(
            f"< {cut:.1f} mm:",
            int((x < cut).sum()),
            f"({(x < cut).mean():.3f})"
        )

out.to_csv(
    "06_Coplanarity/n1_distance_sct_layers.csv",
    index=False
)

print("\nSaved: 06_Coplanarity/n1_distance_sct_layers.csv")
