import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "00_Shared_Code")
from coplanarity import paper_plane_finding_exact8

f = "05_Tracking/converted_photon_hits_weighted_cylinders_eta18_detector_response.csv"
df = pd.read_csv(f)

MAP = {
    1:802, 2:804, 3:806, 4:808,
    5:1302, 6:1304, 7:1306, 8:1308
}

PITCH_S = {1:0.050, 2:0.050, 3:0.050, 4:0.050}
PITCH_Z = {1:0.250, 2:0.400, 3:0.400, 4:0.400}

def merge_event(ev, adjacent=False):
    rows = []

    for layer in range(1, 9):
        g = ev[ev.layer_id == layer].copy()

        if layer > 4 or len(g) < 2:
            rows.extend(g.to_dict("records"))
            continue

        R = float(g["r"].iloc[0])

        g["is"] = np.rint(R * g["phi"] / PITCH_S[layer]).astype(int)
        g["iz"] = np.rint(g["z"] / PITCH_Z[layer]).astype(int)

        used = set()

        for i in range(len(g)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(g)):
                if j in used:
                    continue

                ds = abs(int(g.iloc[i]["is"]) - int(g.iloc[j]["is"]))
                dz = abs(int(g.iloc[i]["iz"]) - int(g.iloc[j]["iz"]))

                if adjacent:
                    do_merge = (ds <= 1 and dz <= 1)
                else:
                    do_merge = (ds == 0 and dz == 0)

                if do_merge:
                    cluster.append(j)
                    used.add(j)

            members = g.iloc[cluster]

            r = members.iloc[0].to_dict()
            r["x"] = members["x"].mean()
            r["y"] = members["y"].mean()
            r["z"] = members["z"].mean()
            r["phi"] = np.arctan2(r["y"], r["x"])

            rows.append(r)

    return pd.DataFrame(rows)

def passes(ev):
    xyz = ev[["x", "y", "z"]].to_numpy(float)
    layers = np.array([MAP[int(x)] for x in ev["layer_id"]], dtype=int)

    return paper_plane_finding_exact8(
        xyz,
        layers
    ).get("found", False)

total = df["event"].nunique()

for label, adjacent in [
    ("SAME PIXEL ONLY", False),
    ("SAME + ADJACENT PIXELS", True)
]:
    found = 0

    for event, ev in df.groupby("event"):
        merged = merge_event(ev, adjacent=adjacent)

        if passes(merged):
            found += 1

    print()
    print(label)
    print("Found:", found)
    print("Acceptance:", found / total)
