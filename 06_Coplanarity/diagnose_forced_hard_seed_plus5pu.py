import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")

from coplanarity import (
    compute_T_tensor,
    _ensure_forward,
)

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_10_hits_detector_response.csv"

PASS_EVENTS = [405, 5833, 8053]
PU_EVENTS = [1,2,3,4,5]

MAP = {
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

for event in PASS_EVENTS:

    h = hard[hard["event"] == event].copy()
    h["source"] = "hard"

    p = pu[pu["pileup_event"].isin(PU_EVENTS)].copy()
    p["source"] = "pileup"

    combo = pd.concat([h, p], ignore_index=True, sort=False)

    xyz = combo[["x","y","z"]].to_numpy(float)
    layers = np.array([MAP[int(x)] for x in combo["layer_id"]], dtype=int)
    sources = combo["source"].to_numpy()

    # genuine hard pair in layer 8
    hard8 = np.where((layers == OUTER) & (sources == "hard"))[0]

    # genuine hard pair in layer 7
    hard7 = np.where((layers == SECOND) & (sources == "hard"))[0]

    print("\n======================================")
    print("EVENT:", event)
    print("hard8 indices:", hard8)
    print("hard7 indices:", hard7)

    if len(hard8) != 2 or len(hard7) != 2:
        print("Unexpected hard-hit multiplicity in seed layers")
        continue

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
        print("Seed tensor invalid")
        continue

    n3 = _ensure_forward(seed_xyz, fit["n3"])

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

        candidates = np.where(layers == lyr)[0]

        layer_acc_hard = 0
        layer_acc_pu = 0
        layer_rej_n1 = 0
        layer_rej_factor = 0

        for cand in candidates:

            if cand in accepted_idx:
                continue

            cf = compute_T_tensor(xyz[accepted_idx])

            n1 = cf["n1"]
            n2 = cf["n2"]
            n3 = cf["n3"]

            if n1 is None or n2 is None or n3 is None:
                continue

            n3 = _ensure_forward(xyz[accepted_idx], n3)

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
            tf = compute_T_tensor(xyz[trial_idx])

            tds = float(tf["delta_s"])
            tdw = float(tf["delta_w"])

            cur = compute_T_tensor(xyz[accepted_idx])
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

    ff = compute_T_tensor(xyz[accepted_idx])

    fds = float(ff["delta_s"])
    fdw = float(ff["delta_w"])

    accepted_layers = layers[accepted_idx]

    counts = {
        lyr: int(np.sum(accepted_layers == lyr))
        for lyr in [802,804,806,808,1302,1304,1306,1308]
    }

    n_layers_ge1 = sum(v >= 1 for v in counts.values())
    n_layers_ge2 = sum(v >= 2 for v in counts.values())

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
    print("coverage pass:", n_layers_ge1 == 8 and n_layers_ge2 >= 7)
