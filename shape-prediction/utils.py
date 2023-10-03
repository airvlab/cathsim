import numpy as np

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


def sample_points(curve, distance=0.002):
    cs_x, cs_y, cs_z, t = curve
    new_points = [np.array([cs_x(t[0]), cs_y(t[0]), cs_z(t[0])])]
    last_point = new_points[0]
    t_current = t[0]

    while t_current < t[-1]:
        t_search = t_current
        dt = 0.001  # Initial step for parameter t

        # Finding t corresponding to the next point at 'distance' from the last point
        while True:
            t_search += dt
            if t_search >= t[-1]:
                # Check the distance to the end point
                end_point = np.array([cs_x(t[-1]), cs_y(t[-1]), cs_z(t[-1])])
                if np.linalg.norm(end_point - last_point) >= distance:
                    new_points.append(end_point)
                break  # End of the curve

            candidate_point = np.array([cs_x(t_search), cs_y(t_search), cs_z(t_search)])
            if np.linalg.norm(candidate_point - last_point) >= distance:
                # Refine t_search for more accurate distance
                for _ in range(10):  # Number of refinement steps
                    dt *= 0.1  # Reduce the step size
                    while np.linalg.norm(candidate_point - last_point) >= distance:
                        t_search -= dt
                        candidate_point = np.array(
                            [cs_x(t_search), cs_y(t_search), cs_z(t_search)]
                        )
                break

        new_points.append(candidate_point)
        last_point = candidate_point
        t_current = t_search

    return np.array(new_points)


def filter_points(
    generated: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(generated.shape[0], actual.shape[0])
    return generated[min_len], actual[min_len]
