import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_sct_rphi_only.csv"

SIGMAS = [0.0, 0.1, 0.3, 0.58, 0.87, 1.16]
SEEDS = list(range(10))

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

base = pd.read_csv(INPUT)
sct_mask = base["layer_id"].isin([5,6,7,8]).to_numpy()

rows = []

for sigma in SIGMAS:

    for seed in SEEDS:

        df = base.copy()

        if sigma > 0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 1.0, sct_mask.sum())

            df.loc[sct_mask, "z"] = (
                base.loc[sct_mask, "z"].to_numpy()
                + sigma * noise
            )

        found = 0
        no_seed = 0

        for event, ev in df.groupby("event"):

            xyz = ev[["x","y","z"]].to_numpy(float)

            layers = np.array(
                [MAP[int(x)] for x in ev["layer_id"]],
                dtype=int
            )

            r = paper_plane_finding_exact8(xyz, layers)

            if r.get("found", False):
                found += 1
            elif r.get("reason") == "no valid outer-layer seed pairs":
                no_seed += 1

        rows.append({
            "sigma_z_mm": sigma,
            "seed": seed,
            "found": found,
            "no_seed": no_seed,
            "acceptance": found / 7303.0
        })

        print(
            f"sigma={sigma:.2f} seed={seed:2d} "
            f"found={found:4d} "
            f"acc={found/7303.0:.6f}"
        )

out = pd.DataFrame(rows)

summary = (
    out.groupby("sigma_z_mm")
       .agg(
           mean_found=("found","mean"),
           std_found=("found","std"),
           min_found=("found","min"),
           max_found=("found","max"),
           mean_acceptance=("acceptance","mean")
       )
       .reset_index()
)

out.to_csv(
    "06_Coplanarity/sct_z_sigma_multiseed_raw.csv",
    index=False
)

summary.to_csv(
    "06_Coplanarity/sct_z_sigma_multiseed_summary.csv",
    index=False
)

print("\n===== SUMMARY =====")
print(summary.to_string(index=False))
