from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_Shared_Code"))

from coplanarity import paper_plane_finding_exact8

INPUT = ROOT / "07_Quirk" / "quirk_hits_methodB_truth.csv"

PIXEL_PITCH_RPHI_MM = {
    1: 0.050,
    2: 0.050,
    3: 0.050,
    4: 0.050,
}

PIXEL_PITCH_Z_MM = {
    1: 0.250,
    2: 0.400,
    3: 0.400,
    4: 0.400,
}

SIGMA_RPHI_MM = 0.034
SIGMA_Z_MM = 1.16

LOGICAL_LAYER = {
    1: 802,
    2: 804,
    3: 806,
    4: 808,
    5: 1302,
    6: 1304,
    7: 1306,
    8: 1308,
}


def pixelize(df):
    out = df.copy()

    for idx in out.index[out["layer_id"].isin([1,2,3,4])]:
        layer = int(out.at[idx, "layer_id"])
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        ps = PIXEL_PITCH_RPHI_MM[layer]
        pz = PIXEL_PITCH_Z_MM[layer]

        s = R * phi
        s = np.round(s / ps) * ps
        z = np.round(z / pz) * pz

        phi = s / R

        out.at[idx, "phi"] = phi
        out.at[idx, "z"] = z
        out.at[idx, "x"] = R * np.cos(phi)
        out.at[idx, "y"] = R * np.sin(phi)

    return out


def smear_sct(df, seed=42):
    out = df.copy()
    rng = np.random.default_rng(seed)

    for idx in out.index[out["layer_id"].isin([5,6,7,8])]:
        R = float(out.at[idx, "r"])
        phi = float(out.at[idx, "phi"])
        z = float(out.at[idx, "z"])

        s = R * phi

        s += rng.normal(0.0, SIGMA_RPHI_MM)
        z += rng.normal(0.0, SIGMA_Z_MM)

        phi = s / R

        out.at[idx, "phi"] = phi
        out.at[idx, "z"] = z
        out.at[idx, "x"] = R * np.cos(phi)
        out.at[idx, "y"] = R * np.sin(phi)

    return out


def direct_tensor(xyz):
    """
    Direct all-hit tensor diagnostic only.
    Same uncentered T = sum x x^T convention.
    """
    if len(xyz) == 0:
        return np.nan, np.nan

    T = xyz.T @ xyz
    vals = np.linalg.eigvalsh(T)
    vals = np.sort(vals)

    N = len(xyz)

    ds = np.sqrt(max(vals[0], 0.0) / N)
    dw = np.sqrt(max(vals[1], 0.0) / N)

    return ds, dw


def evaluate(df):
    q = df.copy()

    q["logical_layer"] = (
        q["layer_id"].astype(int).map(LOGICAL_LAYER)
    )

    xyz = q[["x","y","z"]].to_numpy()
    layers = q["logical_layer"].to_numpy()

    direct_ds, direct_dw = direct_tensor(xyz)

    result = paper_plane_finding_exact8(xyz, layers)

    return {
        "found": bool(result.get("found", False)),
        "delta_s": result.get("delta_s", np.nan),
        "delta_w": result.get("delta_w", np.nan),
        "reason": result.get("reason", ""),
        "direct_ds": direct_ds,
        "direct_dw": direct_dw,
    }


df = pd.read_csv(INPUT)

rows = []

for event_id, g in df.groupby("event"):

    variants = {
        "truth": g.copy(),
        "pixel_only": pixelize(g),
        "sct_only": smear_sct(g, seed=42 + int(event_id)),
        "pixel_plus_sct": smear_sct(
            pixelize(g),
            seed=42 + int(event_id)
        ),
    }

    for name, v in variants.items():
        r = evaluate(v)

        rows.append({
            "event": int(event_id),
            "variant": name,
            **r
        })

out = pd.DataFrame(rows)

print("\n===== PASS COUNTS =====")
summary = (
    out.groupby("variant")["found"]
       .agg(["sum","count","mean"])
)
print(summary)

print("\n===== DIRECT ALL-HIT DELTA_S =====")
print(
    out.groupby("variant")["direct_ds"]
       .agg(["mean","median","min","max"])
)

print("\n===== PER-EVENT RESULTS =====")
print(
    out[
        [
            "event",
            "variant",
            "found",
            "direct_ds",
            "direct_dw",
            "delta_s",
            "delta_w",
            "reason",
        ]
    ].to_string(index=False)
)

OUTPUT = ROOT / "07_Quirk" / "quirk_methodB_response_ablation.csv"
out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
