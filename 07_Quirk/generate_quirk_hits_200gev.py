import pandas as pd

from quirk_trajectory import simulate_quirk_pair_tracks

from quirk_intersections import (
    build_detector_module_table,
    intersect_track_with_modules,
)



# ============================================================
# REAL TRACKML DETECTOR
# ============================================================

DETECTOR_PATH = (
    "/home/rafia/TrackML-note/QuirkTracking-ML/"
    "trackml_Dataset_Detector_WM/trackml_raw/detectors.csv"
)


# ============================================================
# LOAD DETECTOR
# ============================================================

print("Loading detector...")

detector_orig = pd.read_csv(DETECTOR_PATH)

print("Building detector geometry...")

modules = build_detector_module_table(
    detector_orig
)

print(
    "Number of detector modules:",
    len(modules)
)


# ============================================================
# GENERATE QUIRK TRAJECTORY
# ============================================================

print("Generating quirk pair trajectory...")

event = simulate_quirk_pair_tracks(
    event_id=0,

    # trajectory resolution
    n_steps=150000,
    t_max=200.0,

    # detector magnetic field
    b_field=2.0,
    # quirk parameters
    quirk_mass=1800.0,

    # initial production kinematics
    pair_pt=200.0,
    pair_pz=10.0,
    opening_angle=1.3,
    string_tension=2000.0,
)


# ============================================================
# FIND DETECTOR HITS
# ============================================================

print("Finding quirk hits...")


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


# particle labels

if len(hits_q) > 0:
    hits_q["particle"] = 1

if len(hits_aq) > 0:
    hits_aq["particle"] = -1



# ============================================================
# COMBINE
# ============================================================

hits = pd.concat(
    [
        hits_q,
        hits_aq
    ],
    ignore_index=True
)
# ============================================================
# Keep one hit per physical detector layer
# ============================================================

import numpy as np

layer_radii = np.array([
    32, 72, 116, 172,
    260, 360, 500, 660,
    820, 1020
])

hits["layer_r"] = hits["r"].apply(
    lambda x: layer_radii[np.argmin(abs(layer_radii - x))]
)

hits = (
    hits.sort_values("trajectory_step")
        .groupby(["particle", "layer_r"])
        .first()
        .reset_index()
)

hits = hits.drop(columns=["layer_r"], errors="ignore")

# ============================================================
# SAVE
# ============================================================

OUTPUT = "07_Quirk/quirk_hits_200gev.csv"

hits.to_csv(
    OUTPUT,
    index=False
)


print("==============================")
print("Quirk simulation complete")
print("==============================")

print("Saved:", OUTPUT)

print("Total hits:", len(hits))

print(hits.head())
