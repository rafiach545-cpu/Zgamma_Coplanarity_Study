import pandas as pd
import numpy as np

INPUT = "05_Tracking/minbias_1000_hits_cylinders_vertices_eta18z.csv"
OUTPUT = "05_Tracking/minbias_1000_hits_cylinders_vertices_eta18z_beamspot.csv"

SIGMA_Z_MM = 45.0
SEED = 12345

df = pd.read_csv(INPUT)

rng = np.random.default_rng(SEED)

events = np.sort(df["pileup_event"].unique())

# One common primary-interaction z shift per min-bias event
event_z0 = {
    int(ev): float(rng.normal(0.0, SIGMA_Z_MM))
    for ev in events
}

df["vertex_z"] = df["pileup_event"].map(event_z0)

# Preserve Pythia secondary displacement while shifting whole event
df["z"] = df["z"] + df["vertex_z"]
df["zProd"] = df["zProd"] + df["vertex_z"]

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Hits:", len(df))
print("Events:", df["pileup_event"].nunique())
print("Particles:", df["particle_id"].nunique())

print("\n=== EVENT BEAMSPOT Z0 ===")
z0 = pd.Series(event_z0)
print(z0.describe(percentiles=[.01,.05,.5,.95,.99]))

print("\nMean z0:", z0.mean())
print("Std z0:", z0.std())

# Verify exactly one z0 per PU event
print("\nMax unique vertex_z values within one event:",
      df.groupby("pileup_event")["vertex_z"].nunique().max())

print("\nExample event shifts:")
print(z0.head(10))
