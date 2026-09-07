import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import compute_T_tensor, _ensure_forward

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_1000_hits_detector_response.csv"
RES  = "06_Coplanarity/full_mu50_premerge.csv"

EVENTS = [8053]

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

for EVENT in EVENTS:

    row = res[res["event"] == EVENT].iloc[0]

    pu_events = [
        int(x)
        for x in str(row["pu_events"]).split(",")
        if x.strip()
    ]

    h = hard[hard["event"] == EVENT].copy()
    h["source"] = "hard"

    p = pu[pu["pileup_event"].isin(pu_events)].copy()
    p["source"] = "pileup"

    combo = pd.concat([h, p], ignore_index=True, sort=False)

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
    print("Combined hits:", len(combo))
    print("PU events:", pu_events)

    seed_idx = [
        hard8[0], hard8[1],
        hard7[0], hard7[1]
    ]

    seedfit = compute_T_tensor(
        xyz[seed_idx]
    )

    print("\nSEED")
    print("ds:", seedfit["delta_s"])
    print("dw:", seedfit["delta_w"])

    accepted = list(seed_idx)
    accepted_pu = []

    print("\nGROWTH")

    for lyr in GROWTH:

        candidates = np.where(
            layers == lyr
        )[0]

        hard_acc = 0
        pu_acc = 0

        for cand in candidates:

            if cand in accepted:
                continue

            cf = compute_T_tensor(
                xyz[accepted]
            )

            n1 = cf["n1"]
            n2 = cf["n2"]
            n3 = cf["n3"]

            if n1 is None or n2 is None or n3 is None:
                continue

            n3 = _ensure_forward(
                xyz[accepted], n3
            )

            x = xyz[cand]

            if (x @ n3) <= 0:
                continue

            r1 = abs(x @ n1)
            r2 = abs(x @ n2)

            if r1 >= 0.5:
                continue

            if r2 >= 10.0:
                continue

            trial = accepted + [cand]

            tf = compute_T_tensor(
                xyz[trial]
            )

            tds = float(tf["delta_s"])
            tdw = float(tf["delta_w"])

            cds = float(cf["delta_s"])
            cdw = float(cf["delta_w"])

            if cds > 0 and tds > 3.0 * cds:
                continue

            if cdw > 0 and tdw > 3.0 * cdw:
                continue

            accepted.append(cand)

            if sources[cand] == "hard":
                hard_acc += 1
            else:
                pu_acc += 1
                accepted_pu.append(cand)

                print(
                    f"  ACCEPTED PU: layer={lyr} "
                    f"idx={cand} "
                    f"|n1|={r1:.4f} "
                    f"|n2|={r2:.4f} "
                    f"ds {cds:.6f}->{tds:.6f}"
                )

                cols = [
                    "pileup_event",
                    "particle_id",
                    "pdg_id",
                    "charge",
                    "pT",
                    "eta",
                    "layer_id",
                    "x","y","z"
                ]

                print(
                    combo.loc[cand, cols]
                    .to_string()
                )

        print(
            f"layer {lyr}: "
            f"hard accepted={hard_acc}, "
            f"PU accepted={pu_acc}"
        )

    final = compute_T_tensor(
        xyz[accepted]
    )

    accepted_layers = layers[accepted]

    counts = {
        L: int(
            np.sum(accepted_layers == L)
        )
        for L in [
            802,804,806,808,
            1302,1304,1306,1308
        ]
    }

    coverage = (
        sum(v >= 1 for v in counts.values()) == 8
        and
        sum(v >= 2 for v in counts.values()) >= 7
    )

    print("\nFINAL WITH PU")
    print("accepted hits:", len(accepted))
    print("accepted PU:", len(accepted_pu))
    print("ds:", final["delta_s"])
    print("dw:", final["delta_w"])
    print("coverage:", coverage)

    clean_idx = [
        i for i in accepted
        if i not in accepted_pu
    ]

    clean = compute_T_tensor(
        xyz[clean_idx]
    )

    print("\nREMOVE ACCEPTED PU HITS")
    print("hits:", len(clean_idx))
    print("ds:", clean["delta_s"])
    print("dw:", clean["delta_w"])
    print(
        "ds pass:",
        float(clean["delta_s"]) < 0.1
    )
