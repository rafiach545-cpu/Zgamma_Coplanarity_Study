import pandas as pd
import numpy as np

INPUT = "05_Tracking/minbias_1000_hits_cylinders_vertices.csv"
OUTPUT = "05_Tracking/minbias_1000_hits_vertices_beamspot.csv"

SIGMA_Z_MM = 45.0
SEED = 12345

df = pd.read_csv(INPUT)

rng = np.random.default_rng(SEED)
events = np.sort(df["pileup_event"].unique())

event_z0 = {
    int(ev): float(rng.normal(0.0, SIGMA_Z_MM))
    for ev in events
}

# Common interaction-z displacement per pile-up event
df["vertex_z"] = df["pileup_event"].map(event_z0)

# Preserve secondary displacement structure
df["z"] = df["z"] + df["vertex_z"]
df["zProd"] = df["zProd"] + df["vertex_z"]

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Hits:", len(df))
print("Events:", df["pileup_event"].nunique())
print("Particles:", df["particle_id"].nunique())

z0 = pd.Series(event_z0)

print("\nBeamspot mean:", z0.mean())
print("Beamspot std:", z0.std())

print(
    "Max unique vertex_z/event:",
    df.groupby("pileup_event")["vertex_z"].nunique().max()
)

print("\nHits by layer:")
print(df["layer_id"].value_counts().sort_index())
