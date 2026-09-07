from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "00_Shared_Code"))
sys.path.insert(0, str(ROOT / "07_Quirk"))

from quirk_trajectory import simulate_quirk_pair_tracks
from quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules,
)
from coplanarity import paper_plane_finding_exact8


DETECTOR_PATH = (
    "/home/rafia/TrackML-note/QuirkTracking-ML/"
    "trackml_Dataset_Detector_WM/trackml_raw/detectors.csv"
)

print("Loading detector...")
detector_orig = pd.read_csv(DETECTOR_PATH)

print("Building detector geometry...")
modules = build_detector_module_table(detector_orig)

print("Detector modules:", len(modules))

# 16 equally-spaced orientations around full azimuth
phi_values = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)

paper_layers = [802, 804, 806, 808, 1302, 1304, 1306, 1308]

rows = []

for event_id, phi0 in enumerate(phi_values):

    print(f"\nEvent {event_id + 1}/{len(phi_values)}  phi0={phi0:.4f}")

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

    hits_q = intersect_track_with_modules(
        event["xyz_q"],
        modules,
        max_hits=64,
    )

    hits_aq = intersect_track_with_modules(
        event["xyz_aq"],
        modules,
        max_hits=64,
    )

    if len(hits_q):
        hits_q["particle"] = 1

    if len(hits_aq):
        hits_aq["particle"] = -1

    hits = pd.concat([hits_q, hits_aq], ignore_index=True)

    if len(hits) == 0:
        rows.append({
            "event": event_id,
            "phi0": phi0,
            "n_hits_exact8": 0,
            "found": False,
            "delta_s": np.nan,
            "delta_w": np.nan,
            "reason": "no detector hits",
        })
        continue

    hits["physical_layer"] = (
        hits["volume_id"].astype(int) * 100
        + hits["layer_id"].astype(int)
    )

    q = hits[hits["physical_layer"].isin(paper_layers)].copy()

    # Keep first trajectory crossing per particle per physical layer
    if len(q):
        q = (
            q.sort_values("trajectory_step")
             .groupby(["particle", "physical_layer"], as_index=False)
             .first()
        )

    xyz = q[["x", "y", "z"]].to_numpy()
    layers = q["physical_layer"].to_numpy()

    result = paper_plane_finding_exact8(xyz, layers)

    rows.append({
        "event": event_id,
        "phi0": phi0,
        "n_hits_exact8": len(q),
        "found": bool(result.get("found", False)),
        "delta_s": result.get("delta_s", np.nan),
        "delta_w": result.get("delta_w", np.nan),
        "reason": result.get("reason", ""),
    })

df = pd.DataFrame(rows)

out = ROOT / "07_Quirk" / "quirk_phi_scan_exact8.csv"
df.to_csv(out, index=False)

print("\n==============================")
print("Phi scan complete")
print("==============================")
print("Orientations:", len(df))
print("Passed:", int(df["found"].sum()))
print("Pass fraction:", df["found"].mean())
print("Saved:", out)

print("\nResults:")
print(df[["event", "phi0", "n_hits_exact8", "found", "delta_s", "delta_w", "reason"]].to_string(index=False))
