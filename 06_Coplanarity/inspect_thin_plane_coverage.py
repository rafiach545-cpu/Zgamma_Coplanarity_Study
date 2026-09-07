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
    1:802,2:804,3:806,4:808,
    5:1302,6:1304,7:1306,8:1308
}

records = []

for event, ev in df.groupby("event"):

    xyz = ev[["x","y","z"]].to_numpy(float)
    layer_ids = np.array(
        [MAP[int(x)] for x in ev["layer_id"]],
        dtype=int
    )

    outer_pairs = _seed_pairs(
        xyz, layer_ids == OUTER,
        0.10, 20.0, max_pairs=50
    )

    second_pairs = _seed_pairs(
        xyz, layer_ids == SECOND,
        0.10, 20.0, max_pairs=50
    )

    if not outer_pairs or not second_pairs:
        continue

    best = None

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

            cur_ds = float(fit["delta_s"])
            cur_dw = float(fit["delta_w"])

            if cur_ds >= 0.5 or cur_dw >= 10.0:
                continue

            accepted_idx = list(seed_idx)

            for lyr in GROWTH:

                for cand in np.where(layer_ids == lyr)[0]:

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
                        continue
                    if abs(x @ n1) >= 0.5:
                        continue
                    if abs(x @ n2) >= 10.0:
                        continue

                    trial = accepted_idx + [cand]
                    tf = compute_T_tensor(xyz[trial])

                    tds = float(tf["delta_s"])
                    tdw = float(tf["delta_w"])

                    if cur_ds > 0 and tds > 3.0 * cur_ds:
                        continue
                    if cur_dw > 0 and tdw > 3.0 * cur_dw:
                        continue

                    accepted_idx = trial
                    cur_ds = tds
                    cur_dw = tdw

            ff = compute_T_tensor(xyz[accepted_idx])

            ds = float(ff["delta_s"])
            dw = float(ff["delta_w"])

            if ds >= 0.1 or dw >= 10.0:
                continue

            accepted_layers = layer_ids[accepted_idx]

            counts = {
                lyr: int(np.sum(accepted_layers == lyr))
                for lyr in PAPER_LAYERS
            }

            if best is None or ds < best["ds"]:
                best = {
                    "ds": ds,
                    "dw": dw,
                    "counts": counts
                }

    if best is not None:

        counts = best["counts"]

        missing = [
            lyr for lyr in PAPER_LAYERS
            if counts[lyr] == 0
        ]

        one_hit = [
            lyr for lyr in PAPER_LAYERS
            if counts[lyr] == 1
        ]

        n_two = sum(
            counts[lyr] >= 2
            for lyr in PAPER_LAYERS
        )

        if missing:
            status = "missing_layer"
        elif n_two < 7:
            status = "coverage_2hit_fail"
        else:
            status = "pass"

        records.append({
            "event": event,
            "delta_s": best["ds"],
            "delta_w": best["dw"],
            "status": status,
            "missing_layers": ",".join(map(str, missing)),
            "one_hit_layers": ",".join(map(str, one_hit)),
            "n_layers_with_2plus": n_two,
            **{
                f"L{lyr}": counts[lyr]
                for lyr in PAPER_LAYERS
            }
        })

out = pd.DataFrame(records)

print("Thin-plane events (ds<0.1, dw<10):", len(out))

print("\nStatus:")
print(out["status"].value_counts())

print("\nMissing-layer patterns:")
print(
    out.loc[out.status=="missing_layer",
            "missing_layers"]
       .value_counts()
)

print("\nOne-hit layer patterns:")
print(
    out.loc[out.status=="coverage_2hit_fail",
            "one_hit_layers"]
       .value_counts()
)

print("\nAll thin-plane events:")
print(
    out[
        [
            "event","delta_s","delta_w","status",
            "missing_layers","one_hit_layers",
            "n_layers_with_2plus"
        ]
    ].to_string(index=False)
)

out.to_csv(
    "06_Coplanarity/thin_plane_coverage_full_response.csv",
    index=False
)

print(
    "\nSaved:",
    "06_Coplanarity/thin_plane_coverage_full_response.csv"
)
