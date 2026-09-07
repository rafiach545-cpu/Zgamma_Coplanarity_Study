import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")

from coplanarity import (
    compute_T_tensor,
    _seed_pairs,
    _ensure_forward,
)

BASE = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_sct_rphi_only.csv"

SIGMAS = [0.0, 0.1, 0.58, 1.16]

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

base = pd.read_csv(BASE)

sct_mask = base["layer_id"].isin([5,6,7,8]).to_numpy()

rng = np.random.default_rng(42)
noise = np.zeros(len(base))
noise[sct_mask] = rng.normal(0.0, 1.0, sct_mask.sum())

rows = []

for sigma in SIGMAS:

    df = base.copy()

    df.loc[sct_mask, "z"] = (
        base.loc[sct_mask, "z"].to_numpy()
        + sigma * noise[sct_mask]
    )

    vals = {1304: [], 1302: []}

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
                            xyz[accepted_idx], n3
                        )

                        x = xyz[cand]

                        n1dist = abs(float(x @ n1))

                        if lyr in [1304,1302]:
                            vals[lyr].append(n1dist)

                        if (x @ n3) <= 0:
                            continue
                        if n1dist >= 0.5:
                            continue
                        if abs(x @ n2) >= 10.0:
                            continue

                        trial = accepted_idx + [cand]

                        tf = compute_T_tensor(
                            xyz[trial]
                        )

                        tds = float(tf["delta_s"])
                        tdw = float(tf["delta_w"])

                        if cur_ds > 0 and tds > 3.0*cur_ds:
                            continue
                        if cur_dw > 0 and tdw > 3.0*cur_dw:
                            continue

                        accepted_idx = trial
                        cur_ds = tds
                        cur_dw = tdw

    for lyr in [1304,1302]:

        x = np.array(vals[lyr])

        rows.append({
            "sigma_z_mm": sigma,
            "layer": lyr,
            "count": len(x),
            "median_n1_mm": np.median(x),
            "p75_n1_mm": np.quantile(x,0.75),
            "p90_n1_mm": np.quantile(x,0.90),
            "fraction_below_0p5": np.mean(x < 0.5),
        })

out = pd.DataFrame(rows)

print(out.to_string(index=False))

out.to_csv(
    "06_Coplanarity/n1_distance_vs_sigma.csv",
    index=False
)

print("\nSaved: 06_Coplanarity/n1_distance_vs_sigma.csv")
