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

stats = {
    lyr: {
        "tested":0,
        "reject_n3":0,
        "reject_n1":0,
        "reject_n2":0,
        "reject_factor_ds":0,
        "reject_factor_dw":0,
        "accepted":0,
    }
    for lyr in GROWTH
}

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

                    stats[lyr]["tested"] += 1

                    cf = compute_T_tensor(
                        xyz[accepted_idx]
                    )

                    n1 = cf["n1"]
                    n2 = cf["n2"]
                    n3 = cf["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(
                        xyz[accepted_idx], n3
                    )

                    x = xyz[cand]

                    if (x @ n3) <= 0:
                        stats[lyr]["reject_n3"] += 1
                        continue

                    if abs(x @ n1) >= 0.5:
                        stats[lyr]["reject_n1"] += 1
                        continue

                    if abs(x @ n2) >= 10.0:
                        stats[lyr]["reject_n2"] += 1
                        continue

                    trial = accepted_idx + [cand]

                    tf = compute_T_tensor(xyz[trial])

                    tds = float(tf["delta_s"])
                    tdw = float(tf["delta_w"])

                    if cur_ds > 0 and tds > 3.0*cur_ds:
                        stats[lyr]["reject_factor_ds"] += 1
                        continue

                    if cur_dw > 0 and tdw > 3.0*cur_dw:
                        stats[lyr]["reject_factor_dw"] += 1
                        continue

                    accepted_idx = trial
                    cur_ds = tds
                    cur_dw = tdw

                    stats[lyr]["accepted"] += 1

print("===== GROWTH REJECTIONS BY LAYER =====")

for lyr in GROWTH:
    print()
    print("Layer", lyr)

    for key,val in stats[lyr].items():
        print(f"{key}: {val}")
