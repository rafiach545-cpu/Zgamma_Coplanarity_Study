from pathlib import Path
import sys
import io
import contextlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))
sys.path.insert(0, str(ROOT / "07_Quirk"))

with contextlib.redirect_stdout(io.StringIO()):
    import diagnose_methodB_response_components as olddiag

from coplanarity import compute_T_tensor, _seed_pairs, _ensure_forward

pixelize = olddiag.pixelize
LOGICAL_LAYER = olddiag.LOGICAL_LAYER

INPUT = ROOT / "07_Quirk" / "quirk_hits_methodB_truth.csv"

PAPER_LAYERS = [802,804,806,808,1302,1304,1306,1308]
GROWTH_LAYERS = [1304,1302,808,806,804,802]

df = pd.read_csv(INPUT)

rows = []

for event_id, g in df.groupby("event"):

    q = pixelize(g.copy())

    q["logical_layer"] = q["layer_id"].astype(int).map(LOGICAL_LAYER)

    xyz = q[["x","y","z"]].to_numpy(dtype=float)
    layer_ids = q["logical_layer"].to_numpy()

    outer_pairs = _seed_pairs(
        xyz, layer_ids == 1308,
        0.10, 20.0, max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz, layer_ids == 1306,
        0.10, 20.0, max_pairs=50
    )

    # use first seed combination that passes exact seed cuts
    chosen = None

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

            if fit["delta_s"] >= 0.5 or fit["delta_w"] >= 10.0:
                continue

            chosen = seed_idx
            break

        if chosen is not None:
            break

    if chosen is None:
        print("Event", event_id, "has no usable seed")
        continue

    accepted_idx = list(chosen)

    seed_fit = compute_T_tensor(xyz[accepted_idx])
    cur_ds = seed_fit["delta_s"]
    cur_dw = seed_fit["delta_w"]

    for lyr in GROWTH_LAYERS:

        for cand in np.where(layer_ids == lyr)[0]:

            current_fit = compute_T_tensor(xyz[accepted_idx])

            n1 = current_fit["n1"]
            n2 = current_fit["n2"]
            n3 = _ensure_forward(
                xyz[accepted_idx],
                current_fit["n3"]
            )

            x = xyz[cand]

            forward_val = float(x @ n3)
            n1_val = float(abs(x @ n1))
            n2_val = float(abs(x @ n2))

            reason = "accepted"
            trial_ds = np.nan
            trial_dw = np.nan
            ratio_ds = np.nan
            ratio_dw = np.nan

            if forward_val <= 0:
                reason = "forward"
            elif n1_val >= 0.5:
                reason = "n1"
            elif n2_val >= 10.0:
                reason = "n2"
            else:
                trial_idx = accepted_idx + [cand]
                trial_fit = compute_T_tensor(xyz[trial_idx])

                trial_ds = trial_fit["delta_s"]
                trial_dw = trial_fit["delta_w"]

                if cur_ds > 0:
                    ratio_ds = trial_ds / cur_ds

                if cur_dw > 0:
                    ratio_dw = trial_dw / cur_dw

                if cur_ds > 0 and trial_ds > 3.0 * cur_ds:
                    reason = "3x_ds"
                elif cur_dw > 0 and trial_dw > 3.0 * cur_dw:
                    reason = "3x_dw"
                else:
                    accepted_idx.append(cand)
                    cur_ds = trial_ds
                    cur_dw = trial_dw

            rows.append({
                "event": int(event_id),
                "layer": int(lyr),
                "forward_xn3": forward_val,
                "abs_xn1_mm": n1_val,
                "abs_xn2_mm": n2_val,
                "cur_ds_before": cur_ds if reason != "accepted" else np.nan,
                "trial_ds": trial_ds,
                "trial_dw": trial_dw,
                "ds_ratio": ratio_ds,
                "dw_ratio": ratio_dw,
                "result": reason,
            })

out = pd.DataFrame(rows)

print("\n===== PIXEL LAYER REJECTION SUMMARY =====")
pix = out[out["layer"].isin([802,804,806,808])]

print(
    pix.groupby(["layer","result"])
       .size()
       .rename("hits")
       .to_string()
)

print("\n===== PIXEL CANDIDATE DETAILS =====")
print(
    pix[
        [
            "event","layer",
            "forward_xn3",
            "abs_xn1_mm",
            "abs_xn2_mm",
            "trial_ds",
            "ds_ratio",
            "result"
        ]
    ].to_string(index=False)
)

OUTPUT = ROOT / "07_Quirk" / "pixel_growth_rejection_diagnostic.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
