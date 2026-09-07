import numpy as np


# Simplified 8-layer ATLAS-like barrel geometry.
# IMPORTANT:
# These radii are independently sourced ATLAS Run-2 barrel radii.
# They are used as a concrete proxy for the 8-layer geometry described
# by Knapen et al.; the Knapen paper itself does not list these radii.
ATLAS_LAYER_RADII_MM = {
    1: 33.25,   # IBL
    2: 50.5,    # Pixel barrel layer
    3: 88.5,    # Pixel barrel layer
    4: 122.5,   # Pixel barrel layer
    5: 299.0,   # SCT barrel layer
    6: 371.0,   # SCT barrel layer
    7: 443.0,   # SCT barrel layer
    8: 514.0,   # SCT barrel layer
}

# Finite barrel z-extent.
# Exact layer-wise values will be filled only after verification
# from an official ATLAS detector source.
ATLAS_LAYER_ZMAX_MM = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
    8: None,
}
def cylinder_segment_intersection(A, B, r_cyl):
    """
    Exact analytic intersection of a line segment A -> B with an
    infinite cylinder of radius r_cyl around the z-axis.

    Returns a list of (t, point), ordered by increasing t,
    where 0 <= t <= 1.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    d = B - A
    dx = d[0]
    dy = d[1]

    a = dx * dx + dy * dy

    # No transverse motion
    if a < 1e-15:
        return []

    b = 2.0 * (A[0] * dx + A[1] * dy)
    c = A[0] * A[0] + A[1] * A[1] - r_cyl * r_cyl

    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return []

    sqrt_disc = np.sqrt(max(disc, 0.0))

    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    roots = sorted([t1, t2])

    intersections = []

    for t in roots:
        if 0.0 <= t <= 1.0:
            point = A + t * d
            intersections.append((float(t), point))

    return intersections


def intersect_trajectory_with_cylinders(
    traj_xyz,
    radii_dict=ATLAS_LAYER_RADII_MM,
):
    """
    Intersect a charged-particle trajectory with the simplified
    cylindrical detector layers.

    For each layer:
      - scan trajectory in time order
      - require an outward-going segment
      - retain only the first outward crossing

    Therefore each particle can produce at most one hit per layer.
    """
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)

    if len(traj_xyz) < 2:
        return []

    hits = []

    for layer_id, radius_mm in radii_dict.items():

        found = False

        for i in range(len(traj_xyz) - 1):

            A = traj_xyz[i]
            B = traj_xyz[i + 1]

            if not (
                np.all(np.isfinite(A))
                and np.all(np.isfinite(B))
            ):
                continue

            # Outward-going segment only
            rA = np.hypot(A[0], A[1])
            rB = np.hypot(B[0], B[1])

            if rB <= rA:
                continue

            crossings = cylinder_segment_intersection(
                A,
                B,
                radius_mm,
            )

            if not crossings:
                continue

            # Earliest valid crossing on this outward segment
            t, point = crossings[0]

            z_max = ATLAS_LAYER_ZMAX_MM.get(layer_id)

            if z_max is not None:
                if abs(point[2]) > z_max:
                    continue

            hits.append({
                "layer_id": int(layer_id),
                "trajectory_step": int(i),
                "t": float(t),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "r": float(np.hypot(point[0], point[1])),
                "phi": float(np.arctan2(point[1], point[0])),
            })

            found = True
            break

        if not found:
            continue

    return hits
