from pathlib import Path
import sys
import io
import contextlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))
sys.path.insert(0, str(ROOT / "07_Quirk"))

# Reuse EXACT existing detector-response implementation.
# Suppress its top-level diagnostic printing during import.
with contextlib.redirect_stdout(io.StringIO()):
    import diagnose_methodB_response_components as olddiag

pixelize = olddiag.pixelize
smear_sct = olddiag.smear_sct
LOGICAL_LAYER = olddiag.LOGICAL_LAYER

from coplanarity import (
    compute_T_tensor,
    _seed_pairs,
    _ensure_forward,
)

INPUT = ROOT / "07_Quirk" / "quirk_hits_methodB_truth.csv"

PAPER_LAYERS = [802, 804, 806, 808, 1302, 1304, 1306, 1308]
GROWTH_LAYERS = [1304, 1302, 808, 806, 804, 802]


def diagnose_exact8(
    xyz,
    layer_ids,
    dphi_cut=0.10,
    dz_cut_mm=20.0,
    ds_seed=0.5,
    dw_seed=10.0,
    ds_final=0.1,
    dw_final=10.0,
    grow_factor=3.0,
):
    xyz = np.asarray(xyz, dtype=np.float64)
    layer_ids = np.asarray(layer_ids)

    keep = np.isin(layer_ids, PAPER_LAYERS)
    xyz = xyz[keep]
    layer_ids = layer_ids[keep]

    stage = {
        "seed_pairs": False,
        "seed_forward": False,
        "seed_shape": False,
        "growth_started": False,
        "growth_forward": False,
        "growth_n1": False,
        "growth_n2": False,
        "growth_3x": False,
        "final_shape": False,
        "coverage_all8": False,
        "coverage_7double": False,
        "PASS": False,
    }

    best_accepted = 0
    best_counts = {lyr: 0 for lyr in PAPER_LAYERS}
    best_final_ds = np.nan
    best_final_dw = np.nan

    if len(xyz) < 4:
        return stage, "fewer_than_4_hits", best_accepted, best_counts, best_final_ds, best_final_dw

    outer_pairs = _seed_pairs(
        xyz,
        layer_ids == 1308,
        dphi_cut,
        dz_cut_mm,
        max_pairs=50,
    )

    second_pairs = _seed_pairs(
        xyz,
        layer_ids == 1306,
        dphi_cut,
        dz_cut_mm,
        max_pairs=50,
    )

    if not outer_pairs or not second_pairs:
        return stage, "seed_pairs", best_accepted, best_counts, best_final_ds, best_final_dw

    stage["seed_pairs"] = True

    for _, oi, oj in outer_pairs:
        for _, si, sj in second_pairs:

            seed_idx = [oi, oj, si, sj]
            seed_xyz = xyz[seed_idx]

            fit = compute_T_tensor(seed_xyz)

            if fit["n3"] is None:
                continue

            n3 = _ensure_forward(seed_xyz, fit["n3"])

            # Exact strict forward requirement
            if not np.all((seed_xyz @ n3) > 0):
                continue

            stage["seed_forward"] = True

            cur_ds = fit["delta_s"]
            cur_dw = fit["delta_w"]

            if cur_ds >= ds_seed or cur_dw >= dw_seed:
                continue

            stage["seed_shape"] = True

            accepted_idx = list(seed_idx)

            for lyr in GROWTH_LAYERS:

                candidates = np.where(layer_ids == lyr)[0]

                if len(candidates) > 0:
                    stage["growth_started"] = True

                for cand in candidates:

                    if cand in accepted_idx:
                        continue

                    current_fit = compute_T_tensor(xyz[accepted_idx])

                    n1 = current_fit["n1"]
                    n2 = current_fit["n2"]
                    n3 = current_fit["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(xyz[accepted_idx], n3)

                    x = xyz[cand]

                    # forward
                    if (x @ n3) <= 0:
                        continue

                    stage["growth_forward"] = True

                    # plane thickness gate
                    if abs(x @ n1) >= 0.5:
                        continue

                    stage["growth_n1"] = True

                    # plane width gate
                    if abs(x @ n2) >= 10.0:
                        continue

                    stage["growth_n2"] = True

                    trial_idx = accepted_idx + [cand]
                    trial_fit = compute_T_tensor(xyz[trial_idx])

                    t_ds = trial_fit["delta_s"]
                    t_dw = trial_fit["delta_w"]

                    if cur_ds > 0 and t_ds > grow_factor * cur_ds:
                        continue

                    if cur_dw > 0 and t_dw > grow_factor * cur_dw:
                        continue

                    stage["growth_3x"] = True

                    accepted_idx = trial_idx
                    cur_ds = t_ds
                    cur_dw = t_dw

            final_fit = compute_T_tensor(xyz[accepted_idx])

            cur_ds = final_fit["delta_s"]
            cur_dw = final_fit["delta_w"]

            accepted_layers = layer_ids[accepted_idx]

            counts = {
                lyr: int(np.sum(accepted_layers == lyr))
                for lyr in PAPER_LAYERS
            }

            if len(accepted_idx) > best_accepted:
                best_accepted = len(accepted_idx)
                best_counts = counts.copy()
                best_final_ds = cur_ds
                best_final_dw = cur_dw

            if cur_ds >= ds_final or cur_dw >= dw_final:
                continue

            stage["final_shape"] = True

            if not all(counts[lyr] >= 1 for lyr in PAPER_LAYERS):
                continue

            stage["coverage_all8"] = True

            if sum(counts[lyr] >= 2 for lyr in PAPER_LAYERS) < 7:
                continue

            stage["coverage_7double"] = True
            stage["PASS"] = True

    ordered = [
        "seed_pairs",
        "seed_forward",
        "seed_shape",
        "growth_started",
        "growth_forward",
        "growth_n1",
        "growth_n2",
        "growth_3x",
        "final_shape",
        "coverage_all8",
        "coverage_7double",
        "PASS",
    ]

    deepest = "none"
    for name in ordered:
        if stage[name]:
            deepest = name

    return stage, deepest, best_accepted, best_counts, best_final_ds, best_final_dw


df = pd.read_csv(INPUT)

rows = []

for event_id, g in df.groupby("event"):

    variants = {
        "truth": g.copy(),
        "pixel_only": pixelize(g),
        "sct_only": smear_sct(g, seed=42 + int(event_id)),
        "pixel_plus_sct": smear_sct(
            pixelize(g),
            seed=42 + int(event_id)
        ),
    }

    for variant, v in variants.items():

        q = v.copy()
        q["logical_layer"] = (
            q["layer_id"].astype(int).map(LOGICAL_LAYER)
        )

        xyz = q[["x", "y", "z"]].to_numpy()
        layers = q["logical_layer"].to_numpy()

        stage, deepest, nacc, counts, fds, fdw = diagnose_exact8(
            xyz, layers
        )

        row = {
            "event": int(event_id),
            "variant": variant,
            "deepest_stage": deepest,
            "best_naccepted": nacc,
            "best_final_ds": fds,
            "best_final_dw": fdw,
            **stage,
        }

        for lyr in PAPER_LAYERS:
            row[f"n_{lyr}"] = counts.get(lyr, 0)

        rows.append(row)

out = pd.DataFrame(rows)

stage_cols = [
    "seed_pairs",
    "seed_forward",
    "seed_shape",
    "growth_started",
    "growth_forward",
    "growth_n1",
    "growth_n2",
    "growth_3x",
    "final_shape",
    "coverage_all8",
    "coverage_7double",
    "PASS",
]

print("\n===== EXACT8 STAGE COUNTS =====")
print(
    out.groupby("variant")[stage_cols]
       .sum()
       .astype(int)
       .to_string()
)

print("\n===== DEEPEST STAGE COUNTS =====")
print(
    out.groupby(["variant", "deepest_stage"])
       .size()
       .rename("events")
       .to_string()
)

print("\n===== PIXEL-ONLY EVENT DETAILS =====")
cols = [
    "event",
    "deepest_stage",
    "best_naccepted",
    "best_final_ds",
    "best_final_dw",
    "n_802",
    "n_804",
    "n_806",
    "n_808",
    "n_1302",
    "n_1304",
    "n_1306",
    "n_1308",
]
print(
    out[out["variant"] == "pixel_only"][cols]
    .to_string(index=False)
)

OUTPUT = ROOT / "07_Quirk" / "quirk_exact8_stage_diagnostic.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
