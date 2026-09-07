import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")

from coplanarity import (
    compute_T_tensor,
    _ensure_forward,
)

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

row = res[
    (res["event"] == EVENT) &
    (res["mu"] == MU)
].iloc[0]

PU_EVENTS = [
    int(x)
    for x in str(row["pu_events"]).split(",")
    if x.strip()
]

print("PU events:")
print(PU_EVENTS)

h = hard[hard["event"] == EVENT].copy()
h["source"] = "hard"

p = pu[pu["pileup_event"].isin(PU_EVENTS)].copy()
p["source"] = "pileup"

combo = pd.concat(
    [h, p],
    ignore_index=True,
    sort=False
)

xyz = combo[["x","y","z"]].to_numpy(float)
layers = np.array(
    [MAP[int(x)] for x in combo["layer_id"]],
    dtype=int
)
sources = combo["source"].to_numpy()

hard8 = np.where(
    (layers == OUTER) &
    (sources == "hard")
)[0]

hard7 = np.where(
    (layers == SECOND) &
    (sources == "hard")
)[0]

print("\n======================================")
print("EVENT:", EVENT)
print("mu:", MU)
print("Combined hits:", len(combo))
print("hard8 indices:", hard8)
print("hard7 indices:", hard7)

if len(hard8) != 2 or len(hard7) != 2:
    raise RuntimeError(
        "Unexpected hard-hit multiplicity in seed layers"
    )

seed_idx = [
    hard8[0], hard8[1],
    hard7[0], hard7[1]
]

seed_xyz = xyz[seed_idx]

fit = compute_T_tensor(seed_xyz)

ds = float(fit["delta_s"])
dw = float(fit["delta_w"])

print("\nSEED:")
print("delta_s:", ds)
print("delta_w:", dw)

if fit["n3"] is None:
    raise RuntimeError("Seed tensor invalid")

n3 = _ensure_forward(
    seed_xyz,
    fit["n3"]
)

print(
    "forward projections:",
    seed_xyz @ n3
)

print("seed ds < 0.5:", ds < 0.5)
print("seed dw < 10:", dw < 10.0)

accepted_idx = list(seed_idx)

reject_source = {
    "hard": 0,
    "pileup": 0,
}

accept_source = {
    "hard": 0,
    "pileup": 0,
}

print("\nGROWTH:")

for lyr in GROWTH:

    candidates = np.where(
        layers == lyr
    )[0]

    layer_acc_hard = 0
    layer_acc_pu = 0
    layer_rej_n1 = 0
    layer_rej_factor = 0

    for cand in candidates:

        if cand in accepted_idx:
            continue

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

        if (x @ n3) <= 0:
            continue

        if abs(x @ n1) >= 0.5:
            layer_rej_n1 += 1
            reject_source[sources[cand]] += 1
            continue

        if abs(x @ n2) >= 10.0:
            continue

        trial_idx = accepted_idx + [cand]

        tf = compute_T_tensor(
            xyz[trial_idx]
        )

        tds = float(tf["delta_s"])
        tdw = float(tf["delta_w"])

        cur = compute_T_tensor(
            xyz[accepted_idx]
        )

        cds = float(cur["delta_s"])
        cdw = float(cur["delta_w"])

        if cds > 0 and tds > 3.0 * cds:
            layer_rej_factor += 1
            reject_source[sources[cand]] += 1
            continue

        if cdw > 0 and tdw > 3.0 * cdw:
            layer_rej_factor += 1
            reject_source[sources[cand]] += 1
            continue

        accepted_idx.append(cand)
        accept_source[sources[cand]] += 1

        if sources[cand] == "hard":
            layer_acc_hard += 1
        else:
            layer_acc_pu += 1

    print(
        f"layer {lyr}: "
        f"accepted hard={layer_acc_hard}, "
        f"accepted PU={layer_acc_pu}, "
        f"rej_n1={layer_rej_n1}, "
        f"rej_factor={layer_rej_factor}"
    )

ff = compute_T_tensor(
    xyz[accepted_idx]
)

fds = float(ff["delta_s"])
fdw = float(ff["delta_w"])

accepted_layers = layers[
    accepted_idx
]

counts = {
    lyr: int(
        np.sum(
            accepted_layers == lyr
        )
    )
    for lyr in [
        802,804,806,808,
        1302,1304,1306,1308
    ]
}

n_layers_ge1 = sum(
    v >= 1
    for v in counts.values()
)

n_layers_ge2 = sum(
    v >= 2
    for v in counts.values()
)

print("\nFINAL:")
print("accepted hits:", len(accepted_idx))
print("accepted by source:", accept_source)
print("rejected by source:", reject_source)
print("delta_s:", fds)
print("delta_w:", fdw)
print("layer counts:", counts)
print("layers >=1:", n_layers_ge1)
print("layers >=2:", n_layers_ge2)

print("final ds pass:", fds < 0.1)
print("final dw pass:", fdw < 10.0)

coverage_pass = (
    n_layers_ge1 == 8 and
    n_layers_ge2 >= 7
)

print("coverage pass:", coverage_pass)

overall = (
    fds < 0.1 and
    fdw < 10.0 and
    coverage_pass
)

print("\nFORCED HARD-SEED FINAL PASS:", overall)
