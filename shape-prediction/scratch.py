import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from shape_reconstruction import get_points, set_limits, P_TOP, P_SIDE
from cathsim.dm.visualization import point2pixel
from scipy.optimize import minimize_scalar
from scipy import optimize
import cv2

# Load data
DATA_PATH = Path("data/guidewire-reconstruction/")
image_num = 23
points1, points2, points_3d, geom_pos = get_points(image_num)
top = plt.imread(DATA_PATH / f"{image_num}_top.png")
side = plt.imread(DATA_PATH / f"{image_num}_side.png")

iteration = 0


def arc_length(x, y, z):
    """Calculate the arc length of the curve defined by vectors x, y, z."""
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


def get_curve(points_2D):
    """Get cubic spline interpolation of 2D points."""
    t = np.linspace(0, 1, len(points_2D))
    cs_x = CubicSpline(t, points_2D[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, points_2D[:, 1], bc_type="natural")
    return cs_x, cs_y


def resample_curve_equidistant(points, distance=2.0):
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


def get_distance_to_curve(points_2D, curve):
    """Get the distance from each 2D point to the curve defined by cubic splines."""
    cs_x, cs_y = curve
    distances = []

    for point in points_2D:
        # Objective function to minimize
        def objective(t):
            return np.sqrt((cs_x(t) - point[0]) ** 2 + (cs_y(t) - point[1]) ** 2)

        # Minimize the objective function to find closest point on the curve
        res = minimize_scalar(objective, bounds=(0, 1), method="bounded")

        # The minimum distance from the point to the curve
        min_distance = res.fun
        distances.append(min_distance)

    distances = np.array(distances)
    return distances


def fn_loss(points, line):
    return get_distance_to_curve(points, line)


def fn_aggregate(error1, error2):
    return np.sum(error1) + np.sum(error2)


def fn_constraint(points):
    points = points.reshape(-1, 3)
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1) - 0.002)


def objective_function(
    params: np.ndarray,
    line1: np.ndarray,
    line2: np.ndarray,
    fn_loss: callable = fn_loss,
    fn_aggregate: callable = fn_aggregate,
) -> float:
    global iteration

    points_3d = params.reshape(-1, 3)

    projected_points1 = point2pixel(points_3d, P_TOP)
    projected_points2 = point2pixel(points_3d, P_SIDE)

    # overlap points in top and side view
    top_copy = top.copy()
    side_copy = side.copy()
    for point in projected_points1:
        cv2.circle(top_copy, tuple(point.astype(int)), 1, (0, 255, 0), -1)
    for point in projected_points2:
        cv2.circle(side_copy, tuple(point.astype(int)), 1, (0, 255, 0), -1)

    combined = np.concatenate((top_copy, side_copy), axis=1)

    cv2.imshow("combined", combined)
    cv2.waitKey(1)

    error1 = 2 * fn_loss(projected_points1, line1)
    error2 = 2 * fn_loss(projected_points2, line2)

    if fn_aggregate is None:
        total_error = error1 + error2
    else:
        total_error = fn_aggregate(error1, error2)

    # constraint = fn_constraint(points_3d)
    # total_error += 0.2 * constraint

    print(f"Iteration {iteration}: {total_error}", end="\r")
    iteration += 1

    return total_error


def constraint_function(
    params: np.ndarray,
):
    points_3d = params.reshape(-1, 3)
    return np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1) - 0.002)


def optimize_projection(
    points_3d: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    curve1 = get_curve(points1)
    curve2 = get_curve(points2)
    resampled_points = resample_curve_equidistant(points_3d, distance=0.002)

    params_init = resampled_points.flatten()

    params_opt = optimize.minimize(
        objective_function,
        params_init,
        args=(curve1, curve2),
        method="SLSQP",
        options={
            # 'maxiter': 10,
            "disp": True,
            # 'eps': 1e-6,
            # 'gtol': 1e-6,
        },
        constraints={"type": "eq", "fun": constraint_function},
    )

    print(f"Optimization result: {params_opt}")
    points_3d_opt = params_opt.x.reshape(-1, 3)

    return points_3d_opt


if __name__ == "__main__":
    original_top = point2pixel(geom_pos, P_TOP)
    # bound the points to the image
    original_top = original_top[original_top[:, 0] < top.shape[1]]
    original_top = original_top[original_top[:, 1] < top.shape[0]]
    original_top = original_top[original_top[:, 0] > 0]
    original_top = original_top[original_top[:, 1] > 0]

    generated = resample_curve_equidistant(points_3d, distance=0.002)
    reprojected = point2pixel(generated, P_TOP)
    curve = get_curve(points1)
    print(curve)
    # generate the curve

    # distances_original = get_distance_to_curve(points1, curve)
    # distances_reproj = get_distance_to_curve(reprojected, curve)
    reproj_opt = optimize_projection(points_3d, points1, points2)
    reproj_opt_top = point2pixel(reproj_opt, P_TOP)
    reproj_opt_side = point2pixel(reproj_opt, P_SIDE)
    print(reproj_opt.shape)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(top)
    # ax[0].plot(original_top[:, 0], original_top[:, 1], "r.", label="Original")
    # ax[0].plot(reproj_opt_top[:, 0], reproj_opt_top[:, 1], "b.", label="Reprojected")
    # ax[0].set_title("Top view")
    # ax[0].axis("off")
    # ax[1].imshow(side)
    # ax[1].plot(points2[:, 0], points2[:, 1], "r.", label="Original")
    # ax[1].plot(reproj_opt_side[:, 0], reproj_opt_side[:, 1], "b.", label="Reprojected")
    # ax[1].set_title("Side view")
    # ax[1].axis("off")
    #
    # fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.1, 1.1))
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(geom_pos[:, 0], geom_pos[:, 1], geom_pos[:, 2], "r.", label="Original")
    ax.plot(
        reproj_opt[:, 0], reproj_opt[:, 1], reproj_opt[:, 2], "b.", label="Reprojected"
    )
    set_limits(ax)
    plt.show()
