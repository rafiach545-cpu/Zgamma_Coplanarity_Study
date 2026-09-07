import numpy as np
import pandas as pd

INPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_final_response.csv"
OUTPUT = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_beamspot.csv"

df = pd.read_csv(INPUT)
out = df.copy()

BEAMSPOT_SIGMA_Z_MM = 45.0

rng = np.random.default_rng(42)

# One longitudinal beamspot displacement per event.
events = out["event"].drop_duplicates().to_numpy()

event_z0 = {
    event: rng.normal(0.0, BEAMSPOT_SIGMA_Z_MM)
    for event in events
}

out["beamspot_z0"] = out["event"].map(event_z0)
out["z"] = out["z"] + out["beamspot_z0"]

out.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Total hits:", len(out))
print("Events:", out["event"].nunique())
print()
print("Beamspot sigma requested:", BEAMSPOT_SIGMA_Z_MM, "mm")
print("Generated event-z0 mean:", out.drop_duplicates("event")["beamspot_z0"].mean(), "mm")
print("Generated event-z0 std:", out.drop_duplicates("event")["beamspot_z0"].std(), "mm")

print("\nExample event:")
first_event = out["event"].iloc[0]
cols = ["event", "layer_id", "charge", "z", "beamspot_z0", "n_merged"]
print(
    out[out["event"] == first_event][cols]
    .head(12)
    .to_string(index=False)
)
