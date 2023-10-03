import numpy as np

from scipy.interpolate import CubicSpline

from shape_reconstruction import (
    P_TOP,
    P_SIDE,
    get_backbone_points,
    triangulate_points,
    valid_points,
)


def get_points(top: np.ndarray, side: np.ndarray, spacing: int = 15):
    points1 = get_backbone_points(top, spacing=spacing)
    points2 = get_backbone_points(side, spacing=spacing)
    min_length = min(len(points1), len(points2))
    points1 = points1[:min_length]
    points2 = points2[:min_length]

    points_3d_h = triangulate_points(P_TOP, P_SIDE, points1, points2)
    mask = valid_points(points_3d_h)
    if mask is not None:
        points1 = points1[mask]
        points2 = points2[mask]
        points_3d_h = points_3d_h[mask]

    points_3d = points_3d_h[:, :3] / points_3d_h[:, 3, np.newaxis]

    assert (
        points1.shape == points2.shape and points2.shape[0] == points_3d.shape[0]
    ), f"points1.shape = {points1.shape}, points2.shape = {points2.shape}, points_3d.shape = {points_3d.shape}"
    return points_3d


def arc_length(x, y, z):
    """Calculate the arc length of the curve defined by vectors x, y, z."""
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


def sample_points_on_curve(points, distance=2.0):
    total_length = arc_length(points[:, 0], points[:, 1], points[:, 2])

    num_points = int(np.floor(total_length / distance))

    # Create parameter vector
    t = np.linspace(0, 1, len(points))

    # Create cubic splines for x, y, z
    cs_x = CubicSpline(t, points[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, points[:, 1], bc_type="natural")
    cs_z = CubicSpline(t, points[:, 2], bc_type="natural")

    # Initialize variables
    new_points = []
    last_point = points[0]
    t_current = 0
    dt = 0.001  # Step for parameter t

    # Resample curve
    for _ in range(num_points):
        found = False
        while not found:
            t_current += dt
            if t_current > 1.0:
                break  # End of the curve

            candidate_point = np.array(
                [cs_x(t_current), cs_y(t_current), cs_z(t_current)]
            )
            distance_to_last = np.linalg.norm(candidate_point - last_point)

            if distance_to_last >= distance:
                found = True
                new_points.append(candidate_point)
                last_point = candidate_point

    return np.array(new_points)
