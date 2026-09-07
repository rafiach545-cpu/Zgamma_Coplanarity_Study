import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import compute_T_tensor, _ensure_forward

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"
RES  = "06_Coplanarity/pilot_random_mu_premerge_200events.csv"

EVENT = 405
MU = 20

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)
res = pd.read_csv(RES)

row = res[(res["event"] == EVENT) & (res["mu"] == MU)].iloc[0]

pu_events = [
    int(x) for x in str(row["pu_events"]).split(",")
    if x.strip()
]

h = hard[hard["event"] == EVENT].copy()
h["source"] = "hard"

p = pu[pu["pileup_event"].isin(pu_events)].copy()
p["source"] = "pileup"

combo = pd.concat([h, p], ignore_index=True, sort=False)

xyz = combo[["x","y","z"]].to_numpy(float)
layers = np.array([MAP[int(x)] for x in combo["layer_id"]], dtype=int)
sources = combo["source"].to_numpy()

hard8 = np.where((layers == OUTER) & (sources == "hard"))[0]
hard7 = np.where((layers == SECOND) & (sources == "hard"))[0]

accepted = [hard8[0], hard8[1], hard7[0], hard7[1]]
accepted_pu = []

for lyr in GROWTH:

    candidates = np.where(layers == lyr)[0]

    for cand in candidates:

        if cand in accepted:
            continue

        cf = compute_T_tensor(xyz[accepted])

        n1 = cf["n1"]
        n2 = cf["n2"]
        n3 = cf["n3"]

        if n1 is None or n2 is None or n3 is None:
            continue

        n3 = _ensure_forward(xyz[accepted], n3)
        x = xyz[cand]

        if (x @ n3) <= 0:
            continue

        r1 = abs(x @ n1)
        r2 = abs(x @ n2)

        if r1 >= 0.5 or r2 >= 10.0:
            continue

        trial = accepted + [cand]

        tf = compute_T_tensor(xyz[trial])
        cur = compute_T_tensor(xyz[accepted])

        tds = float(tf["delta_s"])
        tdw = float(tf["delta_w"])
        cds = float(cur["delta_s"])
        cdw = float(cur["delta_w"])

        if cds > 0 and tds > 3.0 * cds:
            continue

        if cdw > 0 and tdw > 3.0 * cdw:
            continue

        if sources[cand] == "pileup":
            print("\n=== ACCEPTED PU HIT ===")
            print("combined index:", cand)
            print("logical layer:", lyr)
            print("delta_s before:", cds)
            print("delta_s after :", tds)
            print("delta_w before:", cdw)
            print("delta_w after :", tdw)
            print("|x.n1|:", r1)
            print("|x.n2|:", r2)

            cols = [
                "pileup_event","particle_id","pdg_id",
                "charge","pT","eta","layer_id",
                "x","y","z"
            ]

            print(combo.loc[cand, cols].to_string())
            accepted_pu.append(cand)

        accepted.append(cand)

full = compute_T_tensor(xyz[accepted])

print("\n=== FINAL WITH PU ===")
print("accepted hits:", len(accepted))
print("accepted PU indices:", accepted_pu)
print("delta_s:", full["delta_s"])
print("delta_w:", full["delta_w"])

without_pu = [
    i for i in accepted
    if i not in accepted_pu
]

clean = compute_T_tensor(xyz[without_pu])

print("\n=== SAME ACCEPTED SET, PU HIT REMOVED ===")
print("hits:", len(without_pu))
print("delta_s:", clean["delta_s"])
print("delta_w:", clean["delta_w"])
print("delta_s pass:", float(clean["delta_s"]) < 0.1)
print("delta_w pass:", float(clean["delta_w"]) < 10.0)
