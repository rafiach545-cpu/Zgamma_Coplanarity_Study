from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import paper_plane_finding_exact8


HITS_PATH = (
    ROOT / "05_Tracking" /
    "converted_photon_hits_weighted_cylinders_eta18_beamspot.csv"
)

META_PATH = (
    ROOT / "04_Conversion" /
    "converted_pairs_weighted_barrel_eta18.csv"
)

OUT_PATH = (
    ROOT / "06_Coplanarity" /
    "conversion_coplanarity_weighted_methodB_exact8.csv"
)

BIN_OUT_PATH = (
    ROOT / "06_Coplanarity" /
    "conversion_acceptance_by_pT_weighted_methodB_exact8.csv"
)


hits = pd.read_csv(HITS_PATH)
pairs = pd.read_csv(META_PATH)


# ------------------------------------------------------------
# One metadata row per converted photon/event.
# IMPORTANT: do not count e+ and e- weights separately.
# ------------------------------------------------------------
meta = (
    pairs.sort_values("event")
         .drop_duplicates("event")
         [["event", "photon_pT", "weight"]]
         .copy()
)

print("Metadata events:", len(meta))
print("Hit events:", hits["event"].nunique())


# ------------------------------------------------------------
# Map Method-B cylindrical layers 1..8 onto the logical IDs
# expected by the existing strict exact8 plane finder.
# This changes ONLY layer labels, not hit positions.
# ------------------------------------------------------------
LAYER_MAP = {
    1: 802,
    2: 804,
    3: 806,
    4: 808,
    5: 1302,
    6: 1304,
    7: 1306,
    8: 1308,
}


hits_by_event = {
    event: ev.copy()
    for event, ev in hits.groupby("event")
}


results = []

for i, row in meta.iterrows():

    event = row["event"]
    photon_pT = float(row["photon_pT"])
    weight = float(row["weight"])

    ev = hits_by_event.get(event)

    # Keep event in denominator even if it somehow has no hits.
    if ev is None or len(ev) == 0:

        results.append({
            "event": event,
            "photon_pT": photon_pT,
            "weight": weight,
            "found": False,
            "delta_s": np.nan,
            "delta_w": np.nan,
            "n_hits": 0,
            "reason": "no detector hits",
        })
        continue

    xyz = ev[["x", "y", "z"]].to_numpy(dtype=float)

    layers_methodB = ev["layer_id"].astype(int).to_numpy()

    layers_exact8 = np.array(
        [LAYER_MAP[int(x)] for x in layers_methodB],
        dtype=int
    )

    # Recenter event to its reconstructed primary vertex.
    # The paper plane finder assumes the PV is at the origin.
    vertex_z = float(ev["beamspot_z0"].iloc[0])

    result = paper_plane_finding_exact8(
        xyz,
        layers_exact8,
        vertex_z=vertex_z,
    )

    results.append({
        "event": event,
        "photon_pT": photon_pT,
        "weight": weight,
        "found": bool(result.get("found", False)),
        "delta_s": result.get("delta_s", np.nan),
        "delta_w": result.get("delta_w", np.nan),
        "n_hits": len(ev),
        "reason": result.get("reason", ""),
    })

    if (len(results) % 500) == 0:
        print(f"Processed {len(results)}/{len(meta)} events")


df = pd.DataFrame(results)
df.to_csv(OUT_PATH, index=False)


# ------------------------------------------------------------
# Weighted overall acceptance
# ------------------------------------------------------------
total_weight = df["weight"].sum()
pass_weight = df.loc[df["found"], "weight"].sum()

acceptance = (
    pass_weight / total_weight
    if total_weight > 0
    else np.nan
)


print()
print("Saved:", OUT_PATH)
print("Events:", len(df))
print("Found:", int(df["found"].sum()))
print("Total effective weight:", total_weight)
print("Passed effective weight:", pass_weight)
print("Weighted acceptance:", acceptance)


# ------------------------------------------------------------
# pT bins
# ------------------------------------------------------------
bins = [
    ("<5",     df["photon_pT"] < 5.0),
    ("5-20",  (df["photon_pT"] >= 5.0) &
              (df["photon_pT"] <= 20.0)),
    (">20",    df["photon_pT"] > 20.0),
]

bin_rows = []

for name, mask in bins:

    sub = df.loc[mask].copy()

    n_total = len(sub)
    n_pass = int(sub["found"].sum())

    w_total = sub["weight"].sum()
    w_pass = sub.loc[sub["found"], "weight"].sum()

    acc = (
        w_pass / w_total
        if w_total > 0
        else np.nan
    )

    bin_rows.append({
        "pT_bin": name,
        "n_total": n_total,
        "n_pass": n_pass,
        "effective_total": w_total,
        "effective_pass": w_pass,
        "weighted_acceptance": acc,
    })


bins_df = pd.DataFrame(bin_rows)
bins_df.to_csv(BIN_OUT_PATH, index=False)

print()
print("pT-bin results:")
print(bins_df.to_string(index=False))

print()
print("Saved:", BIN_OUT_PATH)

print()
print("Failure reasons:")
print(
    df.loc[~df["found"], "reason"]
      .value_counts()
      .head(20)
)
