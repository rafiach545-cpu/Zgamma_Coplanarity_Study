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

# ------------------------------------------------------------
# Detector response
# ------------------------------------------------------------

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

phi_values = np.linspace(
    0.0,
    2.0 * np.pi,
    16,
    endpoint=False
)

# Lambda values in eV
lambda_values = [
    1000.0,
    2000.0,
    3000.0,
    5000.0,
    10000.0,
]


def make_hits(traj, particle):
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

    for idx in out.index[out["layer_id"].isin([1,2,3,4])]:
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


def smear_sct(df, rng):
    out = df.copy()

    for idx in out.index[out["layer_id"].isin([5,6,7,8])]:
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        s = R * phi

        s_meas = s + rng.normal(
            0.0,
            SIGMA_RPHI_MM
        )

        z_meas = z + rng.normal(
            0.0,
            SIGMA_Z_MM
        )

        phi_meas = s_meas / R

        out.at[idx, "phi"] = phi_meas
        out.at[idx, "z"] = z_meas
        out.at[idx, "x"] = R * np.cos(phi_meas)
        out.at[idx, "y"] = R * np.sin(phi_meas)

    return out


def merge_pixel_hits(df):

    pixel = df[
        df["layer_id"].isin([1,2,3,4])
    ].copy()

    sct = df[
        df["layer_id"].isin([5,6,7,8])
    ].copy()

    merged_rows = []

    for (event, layer), g in pixel.groupby(
        ["event","layer_id"],
        sort=False
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

            members = [
                records[k]
                for k in cluster
            ]

            x = np.mean([
                m["x"]
                for m in members
            ])

            y = np.mean([
                m["y"]
                for m in members
            ])

            z = np.mean([
                m["z"]
                for m in members
            ])

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
                "phi": np.arctan2(y,x),
                "n_merged": len(members),
            })

    pixel_merged = pd.DataFrame(
        merged_rows
    )

    sct = sct.copy()
    sct["n_merged"] = 1

    out = pd.concat(
        [pixel_merged, sct],
        ignore_index=True
    )

    if len(out):
        out = out.sort_values(
            [
                "event",
                "layer_id",
                "trajectory_step"
            ]
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

    xyz = q[
        ["x","y","z"]
    ].to_numpy()

    layers = q[
        "logical_layer"
    ].to_numpy()

    return paper_plane_finding_exact8(
        xyz,
        layers
    )


summary_rows = []

for lambda_ev in lambda_values:

    lambda_kev = lambda_ev / 1000.0

    print("\n========================================")
    print(
        f"LAMBDA = {lambda_kev:.1f} keV"
    )
    print("========================================")

    # Reset seed for every Lambda:
    # same detector-noise realization pattern
    rng = np.random.default_rng(42)

    truth_rows = []
    response_rows = []

    for event_id, phi0 in enumerate(
        phi_values
    ):

        print(
            f"Lambda={lambda_kev:.1f} keV "
            f"event {event_id+1}/16 "
            f"phi0={phi0:.4f}"
        )

        event = simulate_quirk_pair_tracks(
            event_id=event_id,
            n_steps=150000,
            t_max=200.0,
            b_field=2.0,
            quirk_mass=1800.0,
            pair_pt=250.0,
            pair_pz=10.0,
            opening_angle=1.3,
            phi0=float(phi0),
            string_tension=lambda_ev,
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
            [hits_q,hits_aq],
            ignore_index=True
        )

        truth["event"] = event_id

        # Save raw truth hits for 1800 GeV, Lambda=10 keV, pTpair=250 GeV
        if lambda_ev == 10000.0:
            raw_path = ROOT / "07_Quirk" / "quirk_hits_truth_1800gev_10kev_250gev.csv"

            if event_id == 0:
                truth.to_csv(raw_path, index=False)
            else:
                truth.to_csv(raw_path, mode="a", header=False, index=False)

        truth_result = run_plane_finder(
            truth
        )

        truth_rows.append({
            "event": event_id,
            "phi0": phi0,
            "lambda_eV": lambda_ev,
            "lambda_keV": lambda_kev,
            "n_hits": len(truth),
            "found": bool(
                truth_result.get(
                    "found",
                    False
                )
            ),
            "delta_s": truth_result.get(
                "delta_s",
                np.nan
            ),
            "delta_w": truth_result.get(
                "delta_w",
                np.nan
            ),
            "reason": truth_result.get(
                "reason",
                ""
            ),
        })

        response = pixelize(truth)
        response = smear_sct(
            response,
            rng
        )
        response = merge_pixel_hits(
            response
        )

        response_result = run_plane_finder(
            response
        )

        response_rows.append({
            "event": event_id,
            "phi0": phi0,
            "lambda_eV": lambda_ev,
            "lambda_keV": lambda_kev,
            "n_hits_before_response": len(truth),
            "n_hits_after_response": len(response),
            "hits_removed_by_merge": (
                len(truth)
                - len(response)
            ),
            "found": bool(
                response_result.get(
                    "found",
                    False
                )
            ),
            "delta_s": response_result.get(
                "delta_s",
                np.nan
            ),
            "delta_w": response_result.get(
                "delta_w",
                np.nan
            ),
            "reason": response_result.get(
                "reason",
                ""
            ),
        })

    truth_df = pd.DataFrame(
        truth_rows
    )

    response_df = pd.DataFrame(
        response_rows
    )

    tag = f"{int(lambda_ev)}eV"

    truth_out = (
        ROOT
        / "07_Quirk"
        / f"lambda_scan_truth_{tag}.csv"
    )

    response_out = (
        ROOT
        / "07_Quirk"
        / f"lambda_scan_response_{tag}.csv"
    )

    truth_df.to_csv(
        truth_out,
        index=False
    )

    response_df.to_csv(
        response_out,
        index=False
    )

    truth_pass = int(
        truth_df["found"].sum()
    )

    response_pass = int(
        response_df["found"].sum()
    )

    summary_rows.append({
        "lambda_eV": lambda_ev,
        "lambda_keV": lambda_kev,
        "mass_GeV": 1800.0,
        "n_orientations": 16,
        "truth_pass": truth_pass,
        "truth_efficiency": (
            truth_pass / 16.0
        ),
        "response_pass": response_pass,
        "response_efficiency": (
            response_pass / 16.0
        ),
        "hits_removed_by_merge": int(
            response_df[
                "hits_removed_by_merge"
            ].sum()
        ),
    })

    print(
        f"\nTruth pass: "
        f"{truth_pass}/16"
    )

    print(
        f"Response pass: "
        f"{response_pass}/16"
    )

    print("Saved:")
    print(truth_out)
    print(response_out)


summary = pd.DataFrame(
    summary_rows
)

summary_out = (
    ROOT
    / "07_Quirk"
    / "lambda_scan_methodB_summary.csv"
)

summary.to_csv(
    summary_out,
    index=False
)

print("\n========================================")
print("LAMBDA SCAN COMPLETE")
print("========================================")

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved:")
print(summary_out)
