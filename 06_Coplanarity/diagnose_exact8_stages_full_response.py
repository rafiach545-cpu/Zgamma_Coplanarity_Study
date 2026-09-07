import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")

from coplanarity import (
    compute_T_tensor,
    _seed_pairs,
    _ensure_forward,
)

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"

df = pd.read_csv(INPUT)

PAPER_LAYERS = [802,804,806,808,1302,1304,1306,1308]
OUTER = 1308
SECOND = 1306
GROWTH = [1304,1302,808,806,804,802]

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

DPHI = 0.10
DZ = 20.0

DS_SEED = 0.5
DW_SEED = 10.0

DS_FINAL = 0.1
DW_FINAL = 10.0

GROW_FACTOR = 3.0


global_growth = {
    "tested": 0,
    "reject_n3": 0,
    "reject_n1": 0,
    "reject_n2": 0,
    "reject_factor_ds": 0,
    "reject_factor_dw": 0,
    "accepted": 0,
}


def diagnose_event(ev):

    xyz = ev[["x","y","z"]].to_numpy(float)

    layer_ids = np.array(
        [MAP[int(x)] for x in ev["layer_id"]],
        dtype=int
    )

    outer_pairs = _seed_pairs(
        xyz,
        layer_ids == OUTER,
        DPHI,
        DZ,
        max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz,
        layer_ids == SECOND,
        DPHI,
        DZ,
        max_pairs=50
    )

    if not outer_pairs or not second_pairs:
        return "no_outer_pairs"

    any_forward = False
    any_seed_ds = False
    any_seed_both = False

    any_final_ds = False
    any_final_both = False
    any_all8 = False

    for _, oi, oj in outer_pairs:
        for _, si, sj in second_pairs:

            seed_idx = [oi,oj,si,sj]
            seed_xyz = xyz[seed_idx]

            fit = compute_T_tensor(seed_xyz)

            if fit["n3"] is None:
                continue

            n3 = _ensure_forward(seed_xyz, fit["n3"])

            if not np.all((seed_xyz @ n3) > 0):
                continue

            any_forward = True

            cur_ds = float(fit["delta_s"])
            cur_dw = float(fit["delta_w"])

            if cur_ds >= DS_SEED:
                continue

            any_seed_ds = True

            if cur_dw >= DW_SEED:
                continue

            any_seed_both = True

            accepted_idx = list(seed_idx)

            for lyr in GROWTH:

                candidates = np.where(layer_ids == lyr)[0]

                for cand in candidates:

                    if cand in accepted_idx:
                        continue

                    global_growth["tested"] += 1

                    current_fit = compute_T_tensor(
                        xyz[accepted_idx]
                    )

                    n1 = current_fit["n1"]
                    n2 = current_fit["n2"]
                    n3 = current_fit["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(
                        xyz[accepted_idx],
                        n3
                    )

                    x = xyz[cand]

                    if (x @ n3) <= 0:
                        global_growth["reject_n3"] += 1
                        continue

                    if abs(x @ n1) >= 0.5:
                        global_growth["reject_n1"] += 1
                        continue

                    if abs(x @ n2) >= 10.0:
                        global_growth["reject_n2"] += 1
                        continue

                    trial_idx = accepted_idx + [cand]

                    trial_fit = compute_T_tensor(
                        xyz[trial_idx]
                    )

                    t_ds = float(trial_fit["delta_s"])
                    t_dw = float(trial_fit["delta_w"])

                    if (
                        cur_ds > 0
                        and t_ds > GROW_FACTOR * cur_ds
                    ):
                        global_growth["reject_factor_ds"] += 1
                        continue

                    if (
                        cur_dw > 0
                        and t_dw > GROW_FACTOR * cur_dw
                    ):
                        global_growth["reject_factor_dw"] += 1
                        continue

                    accepted_idx = trial_idx
                    cur_ds = t_ds
                    cur_dw = t_dw

                    global_growth["accepted"] += 1

            final_fit = compute_T_tensor(
                xyz[accepted_idx]
            )

            final_ds = float(final_fit["delta_s"])
            final_dw = float(final_fit["delta_w"])

            if final_ds >= DS_FINAL:
                continue

            any_final_ds = True

            if final_dw >= DW_FINAL:
                continue

            any_final_both = True

            accepted_layers = layer_ids[accepted_idx]

            counts = {
                lyr: int(
                    np.sum(accepted_layers == lyr)
                )
                for lyr in PAPER_LAYERS
            }

            if not all(
                counts[lyr] >= 1
                for lyr in PAPER_LAYERS
            ):
                continue

            any_all8 = True

            if sum(
                counts[lyr] >= 2
                for lyr in PAPER_LAYERS
            ) < 7:
                continue

            return "pass"

    # Furthest stage reached determines failure reason
    if not any_forward:
        return "seed_forward_fail"

    if not any_seed_ds:
        return "seed_ds_fail"

    if not any_seed_both:
        return "seed_dw_fail"

    if not any_final_ds:
        return "final_ds_fail"

    if not any_final_both:
        return "final_dw_fail"

    if not any_all8:
        return "missing_layer_after_growth"

    return "coverage_2hit_fail"


reasons = []

for n, (event, ev) in enumerate(df.groupby("event"), 1):

    reason = diagnose_event(ev)

    reasons.append({
        "event": event,
        "reason": reason
    })

    if n % 500 == 0:
        print(f"Processed {n}/7303")


out = pd.DataFrame(reasons)

print()
print("===== EXACT EVENT-LEVEL STAGE BREAKDOWN =====")
print(out["reason"].value_counts())

print()
print("===== GROWTH HIT-LEVEL REJECTIONS =====")
for k,v in global_growth.items():
    print(f"{k}: {v}")

out.to_csv(
    "06_Coplanarity/full_response_stage_diagnostic.csv",
    index=False
)

print()
print(
    "Saved:",
    "06_Coplanarity/full_response_stage_diagnostic.csv"
)
