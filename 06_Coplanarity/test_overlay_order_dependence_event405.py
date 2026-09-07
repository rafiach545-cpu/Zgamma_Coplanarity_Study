import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

F = "06_Coplanarity/smoke_overlay_event405_plus5pu.csv"

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

df = pd.read_csv(F)

def run(d, label):
    d = d.reset_index(drop=True)

    xyz = d[["x","y","z"]].to_numpy(float)

    layers = np.array(
        [MAP[int(x)] for x in d["layer_id"]],
        dtype=int
    )

    r = paper_plane_finding_exact8(xyz, layers)

    print(
        f"{label:30s}",
        "Found =", r.get("found", False),
        "ds =", r.get("delta_s", None),
        "dw =", r.get("delta_w", None),
        "reason =", r.get("reason", None)
    )

print("Rows:", len(df))

# 1. Current stored ordering
run(df, "CURRENT ORDER")

# 2. Reverse entire dataframe
run(df.iloc[::-1], "REVERSED ORDER")

# 3. Random permutations
for seed in range(10):
    d = df.sample(
        frac=1.0,
        random_state=seed
    )
    run(d, f"RANDOM SEED {seed}")

# 4. Within each layer: hard-containing hits first
if "hard_members" in df.columns:

    d = df.copy()

    d["_hard_first"] = (
        d["hard_members"].fillna(0) > 0
    ).astype(int)

    d = d.sort_values(
        ["layer_id", "_hard_first"],
        ascending=[True, False]
    )

    run(d, "HARD-FIRST PER LAYER")

# 5. Within each layer: pile-up first
if "pileup_members" in df.columns:

    d = df.copy()

    d["_pu_first"] = (
        d["pileup_members"].fillna(0) > 0
    ).astype(int)

    d = d.sort_values(
        ["layer_id", "_pu_first"],
        ascending=[True, False]
    )

    run(d, "PILEUP-FIRST PER LAYER")
