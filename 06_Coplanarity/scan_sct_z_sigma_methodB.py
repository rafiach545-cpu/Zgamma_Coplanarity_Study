import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_sct_rphi_only.csv"

SIGMAS = [0.0, 0.1, 0.3, 0.58, 0.87, 1.16]

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

base = pd.read_csv(INPUT)

# Same random Gaussian draw for every sigma point.
# This makes the scan comparable point-by-point.
rng = np.random.default_rng(42)

sct_mask = base["layer_id"].isin([5,6,7,8])
noise = np.zeros(len(base))
noise[sct_mask.to_numpy()] = rng.normal(
    0.0, 1.0, sct_mask.sum()
)

results = []

for sigma_z in SIGMAS:

    df = base.copy()

    # Rphi smearing is already present in INPUT.
    # Only SCT z smearing is changed here.
    df.loc[sct_mask, "z"] = (
        base.loc[sct_mask, "z"].to_numpy()
        + sigma_z * noise[sct_mask.to_numpy()]
    )

    found = 0
    no_seed = 0
    no_plane = 0

    for event, ev in df.groupby("event"):

        xyz = ev[["x","y","z"]].to_numpy(float)

        layers = np.array(
            [MAP[int(x)] for x in ev["layer_id"]],
            dtype=int
        )

        r = paper_plane_finding_exact8(
            xyz,
            layers
        )

        if r.get("found", False):
            found += 1

        elif r.get("reason") == "no valid outer-layer seed pairs":
            no_seed += 1

        else:
            no_plane += 1

    total = df["event"].nunique()
    acceptance = found / total

    results.append({
        "sigma_z_mm": sigma_z,
        "events": total,
        "found": found,
        "no_seed": no_seed,
        "seed_but_plane_failed": no_plane,
        "acceptance": acceptance
    })

    print(
        f"sigma_z={sigma_z:>4.2f} mm | "
        f"found={found:4d} | "
        f"acceptance={acceptance:.6f} | "
        f"no_seed={no_seed:4d} | "
        f"plane_fail={no_plane:4d}"
    )

out = pd.DataFrame(results)

OUTPUT = "06_Coplanarity/sct_z_sigma_scan_methodB.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("\nFull table:")
print(out.to_string(index=False))
