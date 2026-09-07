from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "00_Shared_Code"))
sys.path.insert(0, str(ROOT / "07_Quirk"))

from quirk_trajectory import simulate_quirk_pair_tracks
from simplified_atlas_geometry import intersect_trajectory_with_cylinders
from coplanarity import paper_plane_finding_exact8


# ============================================================
# Method-B detector response
# ============================================================

PIXEL_PITCH_RPHI_MM = {
    1: 0.050,
    2: 0.050,
    3: 0.050,
    4: 0.050,
}

PIXEL_PITCH_Z_MM = {
    1: 0.250,
    2: 0.400,
    3: 0.400,
    4: 0.400,
}

SIGMA_RPHI_MM = 0.034
SIGMA_Z_MM = 1.16

# Coplanarity code logical layer IDs
LOGICAL_LAYER = {
    1: 802,
    2: 804,
    3: 806,
    4: 808,
    5: 1302,
    6: 1304,
    7: 1306,
    8: 1308,
}

# Reproducible detector smearing
rng = np.random.default_rng(42)


def make_hits(traj, particle):
    """
    Intersect one quirk trajectory with Method-B cylindrical layers.
    """
    rows = intersect_trajectory_with_cylinders(traj)

    out = []
    for h in rows:
        q = dict(h)
        q["particle"] = int(particle)
        q["charge"] = int(particle)
        out.append(q)

    return pd.DataFrame(out)


def pixelize(df):
    out = df.copy()

    for idx in out.index[out["layer_id"].isin([1, 2, 3, 4])]:
        layer = int(out.at[idx, "layer_id"])
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        pitch_s = PIXEL_PITCH_RPHI_MM[layer]
        pitch_z = PIXEL_PITCH_Z_MM[layer]

        s = R * phi

        s_center = np.round(s / pitch_s) * pitch_s
        z_center = np.round(z / pitch_z) * pitch_z

        phi_center = s_center / R

        out.at[idx, "phi"] = phi_center
        out.at[idx, "z"] = z_center
        out.at[idx, "x"] = R * np.cos(phi_center)
        out.at[idx, "y"] = R * np.sin(phi_center)

    return out


def smear_sct(df):
    out = df.copy()

    for idx in out.index[out["layer_id"].isin([5, 6, 7, 8])]:
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        s = R * phi

        s_meas = s + rng.normal(0.0, SIGMA_RPHI_MM)
        z_meas = z + rng.normal(0.0, SIGMA_Z_MM)

        phi_meas = s_meas / R

        out.at[idx, "phi"] = phi_meas
        out.at[idx, "z"] = z_meas
        out.at[idx, "x"] = R * np.cos(phi_meas)
        out.at[idx, "y"] = R * np.sin(phi_meas)

    return out


def merge_pixel_hits(df):
    """
    Same neighboring-pixel rule used for conversion Method-B:
      |delta pix_s| <= 1
      |delta pix_z| <= 1

    Applied transitively within each event/layer.
    """
    pixel = df[df["layer_id"].isin([1, 2, 3, 4])].copy()
    sct = df[df["layer_id"].isin([5, 6, 7, 8])].copy()

    merged_rows = []

    for (event, layer), g in pixel.groupby(
        ["event", "layer_id"], sort=False
    ):
        layer = int(layer)
        R = float(g["r"].iloc[0])

        pitch_s = PIXEL_PITCH_RPHI_MM[layer]
        pitch_z = PIXEL_PITCH_Z_MM[layer]

        g = g.copy()

        g["pix_s"] = np.rint(
            (R * g["phi"]) / pitch_s
        ).astype(int)

        g["pix_z"] = np.rint(
            g["z"] / pitch_z
        ).astype(int)

        records = g.to_dict("records")
        used = [False] * len(records)

        for i in range(len(records)):
            if used[i]:
                continue

            cluster = [i]
            used[i] = True

            changed = True
            while changed:
                changed = False

                for j in range(len(records)):
                    if used[j]:
                        continue

                    for k in cluster:
                        ds = abs(
                            records[j]["pix_s"]
                            - records[k]["pix_s"]
                        )
                        dz = abs(
                            records[j]["pix_z"]
                            - records[k]["pix_z"]
                        )

                        if ds <= 1 and dz <= 1:
                            cluster.append(j)
                            used[j] = True
                            changed = True
                            break

            members = [records[k] for k in cluster]

            x = np.mean([m["x"] for m in members])
            y = np.mean([m["y"] for m in members])
            z = np.mean([m["z"] for m in members])

            merged_rows.append({
                "event": int(event),
                "particle": (
                    members[0]["particle"]
                    if len(members) == 1
                    else 0
                ),
                "charge": (
                    members[0]["charge"]
                    if len(members) == 1
                    else 0
                ),
                "layer_id": layer,
                "trajectory_step": min(
                    m["trajectory_step"]
                    for m in members
                ),
                "x": x,
                "y": y,
                "z": z,
                "r": R,
                "phi": np.arctan2(y, x),
                "n_merged": len(members),
            })

    pixel_merged = pd.DataFrame(merged_rows)

    sct = sct.copy()
    sct["n_merged"] = 1

    out = pd.concat(
        [pixel_merged, sct],
        ignore_index=True
    )

    if len(out):
        out = out.sort_values(
            ["event", "layer_id", "trajectory_step"]
        ).reset_index(drop=True)

    return out


def run_plane_finder(df):
    if len(df) == 0:
        return {
            "found": False,
            "delta_s": np.nan,
            "delta_w": np.nan,
            "reason": "no hits",
        }

    q = df.copy()

    q["logical_layer"] = (
        q["layer_id"]
        .astype(int)
        .map(LOGICAL_LAYER)
    )

    xyz = q[["x", "y", "z"]].to_numpy()
    layers = q["logical_layer"].to_numpy()

    return paper_plane_finding_exact8(
        xyz,
        layers
    )


# ============================================================
# 16 orientations
# ============================================================

phi_values = np.linspace(
    0.0,
    2.0 * np.pi,
    16,
    endpoint=False
)

truth_rows = []
response_rows = []
all_truth_hits = []
all_response_hits = []

for event_id, phi0 in enumerate(phi_values):

    print(
        f"\nEvent {event_id+1}/16 "
        f"phi0={phi0:.4f}"
    )

    event = simulate_quirk_pair_tracks(
        event_id=event_id,
        n_steps=150000,
        t_max=200.0,
        b_field=2.0,
        quirk_mass=1800.0,
        pair_pt=50.0,
        pair_pz=10.0,
        opening_angle=1.3,
        phi0=float(phi0),
        string_tension=2000.0,
    )

    hits_q = make_hits(
        event["xyz_q"],
        particle=1
    )

    hits_aq = make_hits(
        event["xyz_aq"],
        particle=-1
    )

    truth = pd.concat(
        [hits_q, hits_aq],
        ignore_index=True
    )

    truth["event"] = event_id

    # --------------------------------------------------------
    # Truth / ideal Method-B
    # --------------------------------------------------------
    truth_result = run_plane_finder(truth)

    truth_rows.append({
        "event": event_id,
        "phi0": phi0,
        "n_hits": len(truth),
        "found": bool(
            truth_result.get("found", False)
        ),
        "delta_s": truth_result.get(
            "delta_s", np.nan
        ),
        "delta_w": truth_result.get(
            "delta_w", np.nan
        ),
        "reason": truth_result.get(
            "reason", ""
        ),
    })

    all_truth_hits.append(truth)

    # --------------------------------------------------------
    # Detector response
    # --------------------------------------------------------
    response = pixelize(truth)
    response = smear_sct(response)
    response = merge_pixel_hits(response)

    response_result = run_plane_finder(
        response
    )

    response_rows.append({
        "event": event_id,
        "phi0": phi0,
        "n_hits_before_response": len(truth),
        "n_hits_after_response": len(response),
        "hits_removed_by_merge": (
            len(truth) - len(response)
        ),
        "found": bool(
            response_result.get(
                "found", False
            )
        ),
        "delta_s": response_result.get(
            "delta_s", np.nan
        ),
        "delta_w": response_result.get(
            "delta_w", np.nan
        ),
        "reason": response_result.get(
            "reason", ""
        ),
    })

    all_response_hits.append(response)


# ============================================================
# Save
# ============================================================

truth_df = pd.DataFrame(truth_rows)
response_df = pd.DataFrame(response_rows)

truth_hits_df = pd.concat(
    all_truth_hits,
    ignore_index=True
)

response_hits_df = pd.concat(
    all_response_hits,
    ignore_index=True
)

truth_out = (
    ROOT
    / "07_Quirk"
    / "quirk_phi_scan_methodB_truth.csv"
)

response_out = (
    ROOT
    / "07_Quirk"
    / "quirk_phi_scan_methodB_response.csv"
)

truth_hits_out = (
    ROOT
    / "07_Quirk"
    / "quirk_hits_methodB_truth.csv"
)

response_hits_out = (
    ROOT
    / "07_Quirk"
    / "quirk_hits_methodB_response.csv"
)

truth_df.to_csv(
    truth_out,
    index=False
)

response_df.to_csv(
    response_out,
    index=False
)

truth_hits_df.to_csv(
    truth_hits_out,
    index=False
)

response_hits_df.to_csv(
    response_hits_out,
    index=False
)


print("\n========================================")
print("QUIRK METHOD-B VALIDATION COMPLETE")
print("========================================")

print("\nTRUTH")
print("Orientations:", len(truth_df))
print(
    "Passed:",
    int(truth_df["found"].sum())
)
print(
    "Pass fraction:",
    truth_df["found"].mean()
)

print("\nDETECTOR RESPONSE")
print("Orientations:", len(response_df))
print(
    "Passed:",
    int(response_df["found"].sum())
)
print(
    "Pass fraction:",
    response_df["found"].mean()
)

print(
    "Total hits removed by pixel merging:",
    int(
        response_df[
            "hits_removed_by_merge"
        ].sum()
    )
)

print("\nTruth result:")
print(
    truth_df[
        [
            "event",
            "phi0",
            "n_hits",
            "found",
            "delta_s",
            "delta_w",
            "reason",
        ]
    ].to_string(index=False)
)

print("\nDetector-response result:")
print(
    response_df[
        [
            "event",
            "phi0",
            "n_hits_before_response",
            "n_hits_after_response",
            "hits_removed_by_merge",
            "found",
            "delta_s",
            "delta_w",
            "reason",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(truth_out)
print(response_out)
print(truth_hits_out)
print(response_hits_out)
