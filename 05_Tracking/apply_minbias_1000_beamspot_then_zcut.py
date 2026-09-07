import pandas as pd
import numpy as np

INPUT = "05_Tracking/minbias_1000_hits_cylinders_vertices.csv"
OUTPUT = "05_Tracking/minbias_1000_hits_vertices_beamspot_eta18z.csv"

SIGMA_Z_MM = 45.0
SEED = 12345
ETA_MAX = 1.8

R = {
    1: 33.25,
    2: 50.5,
    3: 88.5,
    4: 122.5,
    5: 299.0,
    6: 371.0,
    7: 443.0,
    8: 514.0,
}

df = pd.read_csv(INPUT)

rng = np.random.default_rng(SEED)
events = np.sort(df["pileup_event"].unique())

event_z0 = {
    int(ev): float(rng.normal(0.0, SIGMA_Z_MM))
    for ev in events
}

# Shift whole PU interaction in z
df["vertex_z"] = df["pileup_event"].map(event_z0)
df["z"] = df["z"] + df["vertex_z"]
df["zProd"] = df["zProd"] + df["vertex_z"]

# Apply finite barrel proxy AFTER beamspot displacement
zmax = {L: radius * np.sinh(ETA_MAX) for L, radius in R.items()}
df["zmax_mm"] = df["layer_id"].map(zmax)

keep = df["z"].abs() <= df["zmax_mm"]

out = df.loc[keep].drop(columns=["zmax_mm"]).copy()
out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("\n=== COUNTS ===")
print("Input hits:", len(df))
print("Output hits:", len(out))
print("Removed:", len(df) - len(out))
print("Retention:", len(out) / len(df))

print("\nEvents:", out["pileup_event"].nunique())
print("Particles:", out["particle_id"].nunique())

print("\n=== BEAMSPOT ===")
z0 = pd.Series(event_z0)
print("Mean z0:", z0.mean())
print("Std z0:", z0.std())
print(
    "Max unique z0/event:",
    out.groupby("pileup_event")["vertex_z"].nunique().max()
)

print("\n=== HITS BY LAYER ===")
print(out["layer_id"].value_counts().sort_index())
