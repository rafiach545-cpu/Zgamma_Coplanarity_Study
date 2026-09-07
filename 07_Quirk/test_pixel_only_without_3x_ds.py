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

def run_no_3x_ds(xyz, layer_ids):

    outer_pairs = _seed_pairs(
        xyz, layer_ids == 1308,
        0.10, 20.0, max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz, layer_ids == 1306,
        0.10, 20.0, max_pairs=50
    )

    best = None

    for _, oi, oj in outer_pairs:
        for _, si, sj in second_pairs:

            seed_idx = [oi, oj, si, sj]
            seed_xyz = xyz[seed_idx]

            fit = compute_T_tensor(seed_xyz)

            if fit["n3"] is None:
                continue

            n3 = _ensure_forward(seed_xyz, fit["n3"])

            if not np.all((seed_xyz @ n3) > 0):
                continue

            cur_ds = fit["delta_s"]
            cur_dw = fit["delta_w"]

            if cur_ds >= 0.5 or cur_dw >= 10.0:
                continue

            accepted_idx = list(seed_idx)

            for lyr in GROWTH_LAYERS:

                for cand in np.where(layer_ids == lyr)[0]:

                    current_fit = compute_T_tensor(xyz[accepted_idx])

                    n1 = current_fit["n1"]
                    n2 = current_fit["n2"]
                    n3 = current_fit["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(xyz[accepted_idx], n3)

                    x = xyz[cand]

                    if (x @ n3) <= 0:
                        continue

                    if abs(x @ n1) >= 0.5:
                        continue

                    if abs(x @ n2) >= 10.0:
                        continue

                    trial_idx = accepted_idx + [cand]
                    trial_fit = compute_T_tensor(xyz[trial_idx])

                    t_ds = trial_fit["delta_s"]
                    t_dw = trial_fit["delta_w"]

                    # IMPORTANT:
                    # delta_s 3x rule intentionally disabled here

                    # keep delta_w 3x rule unchanged
                    if cur_dw > 0 and t_dw > 3.0 * cur_dw:
                        continue

                    accepted_idx = trial_idx
                    cur_ds = t_ds
                    cur_dw = t_dw

            final_fit = compute_T_tensor(xyz[accepted_idx])

            ds = final_fit["delta_s"]
            dw = final_fit["delta_w"]

            accepted_layers = layer_ids[accepted_idx]

            counts = {
                lyr: int(np.sum(accepted_layers == lyr))
                for lyr in PAPER_LAYERS
            }

            final_shape = (ds < 0.1 and dw < 10.0)

            coverage_all8 = all(
                counts[lyr] >= 1
                for lyr in PAPER_LAYERS
            )

            coverage_7double = (
                sum(counts[lyr] >= 2 for lyr in PAPER_LAYERS) >= 7
            )

            passed = (
                final_shape
                and coverage_all8
                and coverage_7double
            )

            result = {
                "found": passed,
                "delta_s": ds,
                "delta_w": dw,
                "accepted_hits": len(accepted_idx),
                "coverage_all8": coverage_all8,
                "coverage_7double": coverage_7double,
                **{f"n_{lyr}": counts[lyr] for lyr in PAPER_LAYERS},
            }

            if best is None or ds < best["delta_s"]:
                best = result

            if passed:
                return result

    return best


for event_id, g in df.groupby("event"):

    q = pixelize(g.copy())

    q["logical_layer"] = (
        q["layer_id"].astype(int).map(LOGICAL_LAYER)
    )

    xyz = q[["x","y","z"]].to_numpy(dtype=float)
    layers = q["logical_layer"].to_numpy()

    r = run_no_3x_ds(xyz, layers)

    rows.append({
        "event": int(event_id),
        **r
    })

out = pd.DataFrame(rows)

print("\n===== PIXEL-ONLY WITHOUT 3x DELTA_S RULE =====")
print(
    out[
        [
            "event",
            "found",
            "delta_s",
            "delta_w",
            "accepted_hits",
            "coverage_all8",
            "coverage_7double",
        ]
    ].to_string(index=False)
)

print("\nPASS COUNT:")
print(
    int(out["found"].sum()),
    "/",
    len(out)
)

print("\nLAYER COUNTS:")
print(
    out[
        [
            "event",
            "n_802",
            "n_804",
            "n_806",
            "n_808",
            "n_1302",
            "n_1304",
            "n_1306",
            "n_1308",
        ]
    ].to_string(index=False)
)

OUTPUT = ROOT / "07_Quirk" / "pixel_only_no_3x_ds_ablation.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
