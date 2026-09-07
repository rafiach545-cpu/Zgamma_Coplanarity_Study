import sys
import numpy as np
import pandas as pd
from itertools import combinations, product

sys.path.insert(0, "00_Shared_Code")
from coplanarity import compute_T_tensor

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"

df = pd.read_csv(INPUT)

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

def dphi(a, b):
    d = abs(a - b)
    return min(d, 2*np.pi - d)

def valid_pairs(ev, layer):
    g = ev[ev["layer_id"] == layer]
    out = []

    for i, j in combinations(g.index, 2):
        a = g.loc[i]
        b = g.loc[j]

        if dphi(a["phi"], b["phi"]) < 0.1 and abs(a["z"] - b["z"]) < 20.0:
            out.append((i, j))

    return out

no_outer_pair = 0
seed_pair_exists = 0
seed_ds_fail = 0
seed_dw_fail = 0
seed_pass = 0

seed_ds_values = []
seed_dw_values = []

for event, ev in df.groupby("event"):

    p8 = valid_pairs(ev, 8)
    p7 = valid_pairs(ev, 7)

    if not p8 or not p7:
        no_outer_pair += 1
        continue

    seed_pair_exists += 1

    best = None

    for a, b in p8:
        for c, d in p7:

            xyz = ev.loc[[a,b,c,d], ["x","y","z"]].to_numpy(float)

            fit = compute_T_tensor(xyz)

            ds = float(fit["delta_s"])
            dw = float(fit["delta_w"])

            seed_ds_values.append(ds)
            seed_dw_values.append(dw)

            if best is None or ds < best[0]:
                best = (ds, dw)

    ds, dw = best

    if ds >= 0.5:
        seed_ds_fail += 1
    elif dw >= 10.0:
        seed_dw_fail += 1
    else:
        seed_pass += 1

print("===== FULL SCT RESPONSE SEED DIAGNOSTIC =====")
print("Total events:", df["event"].nunique())
print("No valid outer-layer pair:", no_outer_pair)
print("Valid outer pairs exist:", seed_pair_exists)
print("Best seed fails Delta_s >= 0.5 mm:", seed_ds_fail)
print("Best seed fails Delta_w >= 10 mm:", seed_dw_fail)
print("Best seed passes ds and dw:", seed_pass)

if seed_ds_values:
    x = np.array(seed_ds_values)
    print()
    print("All candidate seed Delta_s:")
    print("median:", np.median(x))
    print("90%:", np.quantile(x, 0.90))
    print("99%:", np.quantile(x, 0.99))

if seed_dw_values:
    x = np.array(seed_dw_values)
    print()
    print("All candidate seed Delta_w:")
    print("median:", np.median(x))
    print("90%:", np.quantile(x, 0.90))
    print("99%:", np.quantile(x, 0.99))
