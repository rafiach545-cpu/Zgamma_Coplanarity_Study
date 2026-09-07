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

INPUT = ROOT / "07_Quirk" / "quirk_hits_methodB_truth_250gev.csv"

SIGMA_RPHI_MM = 0.034
SIGMA_Z_MM = 1.16
DS_FLOOR_MM = 0.020

# Multiple detector-noise replicas
N_TRIALS = 20

PAPER_LAYERS = [802,804,806,808,1302,1304,1306,1308]
GROWTH_LAYERS = [1304,1302,808,806,804,802]

df = pd.read_csv(INPUT)


def smear_sct_matched(df, event_id, trial_id, mode):
    """
    Matched random numbers:
      same drphi used in rphi_only and full
      same dz    used in z_only   and full
    """
    out = df.copy()

    # deterministic but different for every event/trial
    seed = 100000 + 1000 * int(trial_id) + int(event_id)
    rng = np.random.default_rng(seed)

    sct_idx = out.index[out["layer_id"].isin([5,6,7,8])]

    # Generate both sets regardless of mode,
    # so random-number matching is preserved.
    drphi = rng.normal(0.0, SIGMA_RPHI_MM, len(sct_idx))
    dz    = rng.normal(0.0, SIGMA_Z_MM, len(sct_idx))

    for j, idx in enumerate(sct_idx):
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        s = R * phi

        if mode in ("rphi_only", "full"):
            s += drphi[j]

        if mode in ("z_only", "full"):
            z += dz[j]

        phi = s / R

        out.at[idx, "phi"] = phi
        out.at[idx, "z"] = z
        out.at[idx, "x"] = R * np.cos(phi)
        out.at[idx, "y"] = R * np.sin(phi)

    return out


def run_exact8_floor(xyz, layer_ids):

    outer_pairs = _seed_pairs(
        xyz, layer_ids == 1308,
        0.10, 20.0, max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz, layer_ids == 1306,
        0.10, 20.0, max_pairs=50
    )

    # Furthest-stage bookkeeping across ALL seed combinations
    any_forward = False
    any_seed_ds = False
    any_seed_both = False
    any_final_ds = False
    any_final_both = False
    any_all8 = False

    # Keep representative numerical result as before:
    # lowest final delta_s candidate.
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

            any_forward = True

            cur_ds = float(fit["delta_s"])
            cur_dw = float(fit["delta_w"])

            # Original seed ds cut unchanged
            if cur_ds >= 0.5:
                continue

            any_seed_ds = True

            # Original seed dw cut unchanged
            if cur_dw >= 10.0:
                continue

            any_seed_both = True

            accepted_idx = list(seed_idx)

            for lyr in GROWTH_LAYERS:

                for cand in np.where(layer_ids == lyr)[0]:

                    if cand in accepted_idx:
                        continue

                    current_fit = compute_T_tensor(
                        xyz[accepted_idx]
                    )

                    n1 = current_fit["n1"]
                    n2 = current_fit["n2"]
                    n3 = current_fit["n3"]

                    if n1 is None or n2 is None or n3 is None:
                        continue

                    n3 = _ensure_forward(
                        xyz[accepted_idx], n3
                    )

                    x = xyz[cand]

                    # Original growth geometry cuts unchanged
                    if (x @ n3) <= 0:
                        continue

                    if abs(x @ n1) >= 0.5:
                        continue

                    if abs(x @ n2) >= 10.0:
                        continue

                    trial_idx = accepted_idx + [cand]

                    trial_fit = compute_T_tensor(
                        xyz[trial_idx]
                    )

                    t_ds = float(trial_fit["delta_s"])
                    t_dw = float(trial_fit["delta_w"])

                    # Diagnostic ds floor:
                    # ONLY stabilizes the 3x growth comparison.
                    effective_ds = max(
                        cur_ds,
                        DS_FLOOR_MM
                    )

                    if t_ds > 3.0 * effective_ds:
                        continue

                    # Delta_w 3x rule unchanged
                    if (
                        cur_dw > 0
                        and t_dw > 3.0 * cur_dw
                    ):
                        continue

                    accepted_idx = trial_idx
                    cur_ds = t_ds
                    cur_dw = t_dw

            final_fit = compute_T_tensor(
                xyz[accepted_idx]
            )

            ds = float(final_fit["delta_s"])
            dw = float(final_fit["delta_w"])

            accepted_layers = layer_ids[accepted_idx]

            counts = {
                lyr: int(
                    np.sum(accepted_layers == lyr)
                )
                for lyr in PAPER_LAYERS
            }

            all8 = all(
                counts[lyr] >= 1
                for lyr in PAPER_LAYERS
            )

            seven_double = (
                sum(
                    counts[lyr] >= 2
                    for lyr in PAPER_LAYERS
                ) >= 7
            )

            candidate = {
                "found": False,
                "delta_s": ds,
                "delta_w": dw,
                "accepted_hits": len(accepted_idx),
                "coverage_all8": all8,
                "coverage_7double": seven_double,
            }

            if best is None or ds < best["delta_s"]:
                best = candidate.copy()

            # Final ds cut unchanged
            if ds >= 0.1:
                continue

            any_final_ds = True

            # Final dw cut unchanged
            if dw >= 10.0:
                continue

            any_final_both = True

            if not all8:
                continue

            any_all8 = True

            if not seven_double:
                continue

            candidate["found"] = True
            candidate["reason"] = "pass"
            return candidate

    # Assign failure by furthest stage reached
    if not any_forward:
        reason = "seed_forward_fail"
    elif not any_seed_ds:
        reason = "seed_ds_fail"
    elif not any_seed_both:
        reason = "seed_dw_fail"
    elif not any_final_ds:
        reason = "final_ds_fail"
    elif not any_final_both:
        reason = "final_dw_fail"
    elif not any_all8:
        reason = "missing_layer_after_growth"
    else:
        reason = "coverage_2hit_fail"

    if best is None:
        return {
            "found": False,
            "reason": reason,
            "delta_s": np.nan,
            "delta_w": np.nan,
            "accepted_hits": 0,
            "coverage_all8": False,
            "coverage_7double": False,
        }

    best["reason"] = reason
    return best


rows = []

MODES = [
    "pixel_only",
    "rphi_only",
    "z_only",
    "full",
]

for trial in range(N_TRIALS):

    for event_id, g in df.groupby("event"):

        # Pixel response always included first,
        # because this is the realistic quirk detector-response chain.
        pix = pixelize(g.copy())

        variants = {
            "pixel_only": pix.copy(),
            "rphi_only": smear_sct_matched(
                pix, event_id, trial, "rphi_only"
            ),
            "z_only": smear_sct_matched(
                pix, event_id, trial, "z_only"
            ),
            "full": smear_sct_matched(
                pix, event_id, trial, "full"
            ),
        }

        for mode, q in variants.items():

            q = q.copy()

            q["logical_layer"] = (
                q["layer_id"]
                .astype(int)
                .map(LOGICAL_LAYER)
            )

            xyz = q[["x","y","z"]].to_numpy(
                dtype=float
            )

            layers = q["logical_layer"].to_numpy()

            result = run_exact8_floor(
                xyz, layers
            )

            rows.append({
                "trial": trial,
                "event": int(event_id),
                "mode": mode,
                **result,
            })


out = pd.DataFrame(rows)

print("\n===== SCT COORDINATE ABLATION =====")
print("DS floor =", DS_FLOOR_MM, "mm")
print("Trials =", N_TRIALS)
print("Events per trial = 16")

summary = (
    out.groupby("mode")
       .agg(
           passed=("found", "sum"),
           total=("found", "count"),
           mean_ds=("delta_s", "mean"),
           median_ds=("delta_s", "median"),
           min_ds=("delta_s", "min"),
           max_ds=("delta_s", "max"),
           mean_hits=("accepted_hits", "mean"),
       )
)

summary["pass_fraction"] = (
    summary["passed"] / summary["total"]
)

print("\n===== OVERALL SUMMARY =====")
print(summary.to_string())

print("\n===== PASS FRACTION PER TRIAL =====")

trial_summary = (
    out.groupby(["trial","mode"])["found"]
       .mean()
       .unstack()
)

print(trial_summary.to_string())

print("\n===== DELTA_S PER MODE =====")

print(
    out.groupby("mode")["delta_s"]
       .describe()
       .to_string()
)

OUTPUT = (
    ROOT /
    "07_Quirk" /
    "sct_coordinate_failure_diagnostic_250gev.csv"
)

out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
