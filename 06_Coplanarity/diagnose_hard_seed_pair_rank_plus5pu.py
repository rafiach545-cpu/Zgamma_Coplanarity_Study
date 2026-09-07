import numpy as np
import pandas as pd

HARD = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
PU   = "05_Tracking/minbias_10_hits_detector_response.csv"

PASS_EVENTS = [405, 5833, 8053]
PU_EVENTS = [1,2,3,4,5]

DPHI_CUT = 0.10
DZ_CUT = 20.0
CAP = 50

hard = pd.read_csv(HARD)
pu = pd.read_csv(PU)

def pair_list(g):
    """
    Return all valid pairs sorted by smallest dphi,
    matching _seed_pairs logic, but WITHOUT max_pairs cap.
    """
    g = g.reset_index(drop=True)

    phi = np.arctan2(
        g["y"].to_numpy(float),
        g["x"].to_numpy(float)
    )

    z = g["z"].to_numpy(float)

    pairs = []

    for i in range(len(g)):
        for j in range(i+1, len(g)):

            dphi = abs(phi[i] - phi[j])

            if dphi > np.pi:
                dphi = 2*np.pi - dphi

            dz = abs(z[i] - z[j])

            if dphi < DPHI_CUT and dz < DZ_CUT:

                pairs.append({
                    "i": i,
                    "j": j,
                    "dphi": dphi,
                    "dz": dz,
                    "source_i": g.loc[i, "source"],
                    "source_j": g.loc[j, "source"],
                })

    pairs.sort(key=lambda x: x["dphi"])

    return pairs


rows = []

for event in PASS_EVENTS:

    h = hard[hard["event"] == event].copy()
    h["source"] = "hard"

    p = pu[
        pu["pileup_event"].isin(PU_EVENTS)
    ].copy()

    p["source"] = "pileup"

    combo = pd.concat(
        [h, p],
        ignore_index=True,
        sort=False
    )

    print("\n====================================")
    print("HARD EVENT:", event)

    for layer in [8,7]:

        g = combo[
            combo["layer_id"] == layer
        ].copy()

        pairs = pair_list(g)

        hard_pairs = [
            (rank, x)
            for rank, x in enumerate(pairs, start=1)
            if x["source_i"] == "hard"
            and x["source_j"] == "hard"
        ]

        print(f"\nLayer {layer}")
        print("Hits:", len(g))
        print("All valid seed pairs:", len(pairs))

        if not hard_pairs:
            print("Genuine hard-hard pair: NOT VALID under dphi/dz cuts")

            rows.append({
                "event": event,
                "layer": layer,
                "hits": len(g),
                "valid_pairs": len(pairs),
                "hard_pair_exists": False,
                "hard_pair_rank": np.nan,
                "hard_pair_dphi": np.nan,
                "hard_pair_dz": np.nan,
                "inside_top50": False,
            })

            continue

        # normally there should be exactly one hard-hard pair
        rank, hp = hard_pairs[0]

        print("Hard pair rank:", rank)
        print("Hard pair dphi:", hp["dphi"])
        print("Hard pair dz:", hp["dz"])
        print("Inside top 50:", rank <= CAP)

        rows.append({
            "event": event,
            "layer": layer,
            "hits": len(g),
            "valid_pairs": len(pairs),
            "hard_pair_exists": True,
            "hard_pair_rank": rank,
            "hard_pair_dphi": hp["dphi"],
            "hard_pair_dz": hp["dz"],
            "inside_top50": rank <= CAP,
        })

out = pd.DataFrame(rows)

print("\n\n===== SUMMARY =====")
print(out.to_string(index=False))

out.to_csv(
    "06_Coplanarity/hard_seed_pair_rank_plus5pu.csv",
    index=False
)

print("\nSaved:")
print("06_Coplanarity/hard_seed_pair_rank_plus5pu.csv")
