import numpy as np
import pandas as pd

INPUT = "05_Tracking/minbias_10_hits_cylinders.csv"
OUTPUT = "05_Tracking/minbias_10_hits_cylinders_beamspot.csv"

df = pd.read_csv(INPUT)

rng = np.random.default_rng(12345)

events = sorted(df["pileup_event"].unique())

z0_map = {
    ev: rng.normal(0.0, 45.0)
    for ev in events
}

df["pileup_z0"] = df["pileup_event"].map(z0_map)

# Whole pp interaction translated by one common primary z vertex
df["z"] = df["z"] + df["pileup_z0"]

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)

print("\nPile-up event z0 [mm]:")
for ev in events:
    print(ev, z0_map[ev])

print("\nBeamspot z0 summary:")
print(
    pd.Series(list(z0_map.values())).describe()
)

print("\nRows:", len(df))
print("Events:", df["pileup_event"].nunique())
