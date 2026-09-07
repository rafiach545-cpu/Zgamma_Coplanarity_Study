from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import paper_plane_finding_exact8

META = (
    pd.read_csv(
        ROOT / "04_Conversion/converted_pairs_weighted_paper10k_postshower_barrel_eta18.csv"
    )
    .sort_values("event")
    .drop_duplicates("event")
)

PRE = pd.read_csv(
    ROOT / "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_postshower_eta18_detector_response_beamspot.csv"
)

POST = pd.read_csv(
    ROOT / "05_Tracking/converted_photon_hits_weighted_cylinders_paper10k_postshower_eta18_final_response_beamspot.csv"
)

LAYER_MAP = {
    1: 802, 2: 804, 3: 806, 4: 808,
    5: 1302, 6: 1304, 7: 1306, 8: 1308
}

def run(df, label, outfile):
    by_event = {e: g for e, g in df.groupby("event")}
    rows = []

    for n, row in enumerate(META.itertuples(index=False), 1):
        ev = row.event
        g = by_event.get(ev)

        if g is None or len(g) == 0:
            rows.append({
                "event": ev,
                "photon_pT": row.photon_pT,
                "weight": row.weight,
                "found": False,
                "delta_s": np.nan,
                "delta_w": np.nan,
                "n_hits": 0,
                "reason": "no detector hits",
            })
            continue

        xyz = g[["x", "y", "z"]].to_numpy(float)

        layers = np.array(
            [LAYER_MAP[int(x)] for x in g["layer_id"]],
            dtype=int
        )

        z0 = float(g["beamspot_z0"].iloc[0])

        result = paper_plane_finding_exact8(
            xyz,
            layers,
            vertex_z=z0,
        )

        rows.append({
            "event": ev,
            "photon_pT": row.photon_pT,
            "weight": row.weight,
            "found": bool(result.get("found", False)),
            "delta_s": result.get("delta_s", np.nan),
            "delta_w": result.get("delta_w", np.nan),
            "n_hits": len(g),
            "reason": result.get("reason", ""),
        })

        if n % 500 == 0:
            print(f"{label}: processed {n}/{len(META)}")

    out = pd.DataFrame(rows)
    out.to_csv(outfile, index=False)

    n_total = len(out)
    n_pass = int(out["found"].sum())

    print()
    print(f"=== {label} FINAL ===")
    print("Events:", n_total)
    print("Found:", n_pass)
    print("Acceptance:", n_pass / n_total)
    print("Effective total weight:", out["weight"].sum())
    print("Effective pass weight:", out.loc[out["found"], "weight"].sum())

    print("\nTop failure reasons:")
    print(
        out.loc[~out["found"], "reason"]
        .value_counts()
        .head(10)
    )

    print("\nSaved:", outfile)

run(
    PRE,
    "PRE-MERGE",
    ROOT / "06_Coplanarity/paper10k_postshower_noPU_premerge_exact8_corrected.csv",
)

run(
    POST,
    "POST-MERGE",
    ROOT / "06_Coplanarity/paper10k_postshower_noPU_postmerge_exact8_corrected.csv",
)
