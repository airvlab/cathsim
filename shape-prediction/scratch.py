import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import matplotlib.pyplot as plt
from shape_reconstruction import set_limits, P_TOP, P_SIDE
from cathsim.dm.visualization import point2pixel
from scipy import optimize
import cv2

from utils import get_points, sample_points_on_curve

# Load data
DATA_PATH = Path("data/guidewire-reconstruction/")
image_num = 50
points1, points2, points_3d, geom_pos = get_points(image_num, spacing=4)
top = plt.imread(DATA_PATH / f"{image_num}_top.png")
side = plt.imread(DATA_PATH / f"{image_num}_side.png")

iteration = 0


def arc_length(x, y, z):
    """Calculate the arc length of the curve defined by vectors x, y, z."""
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))


def get_curve_2d(points_2D, s=1.0):
    """Get smooth spline interpolation of 2D points."""
    # Compute the cumulative arc length as the parameter
    t = np.zeros(len(points_2D))
    for i in range(1, len(points_2D)):
        t[i] = t[i - 1] + np.linalg.norm(points_2D[i] - points_2D[i - 1])

    # Create smoothing splines for x, y
    cs_x = UnivariateSpline(t, points_2D[:, 0], s=s)
    cs_y = UnivariateSpline(t, points_2D[:, 1], s=s)

    return cs_x, cs_y, t


def plot_curve_on_image(ax, curve):
    cs_x, cs_y, t = curve
    t_new = np.linspace(0, t[-1], 1000)
    curve_points_x = cs_x(t_new)
    curve_points_y = cs_y(t_new)

    ax.plot(curve_points_x, curve_points_y, "r-", linewidth=1)


def get_distance_to_curve(points_2D, curve):
    cs_x, cs_y, t = curve
    t_new = np.linspace(0, t[-1], 1000)
    curve_points = np.vstack((cs_x(t_new), cs_y(t_new))).T

    distances = []
    for point_2D in points_2D:
        distance_array = np.linalg.norm(curve_points - point_2D, axis=1)
        distances.append(np.min(distance_array))
    return distances


def get_curve_3D(points, s=1.0):
    """Get smooth spline interpolation of 3D points."""
    # Compute the cumulative arc length as the parameter
    t = np.zeros(len(points))
    for i in range(1, len(points)):
        t[i] = t[i - 1] + np.linalg.norm(points[i] - points[i - 1])

    # Create smoothing splines for x, y, z
    cs_x = UnivariateSpline(t, points[:, 0], s=s)
    cs_y = UnivariateSpline(t, points[:, 1], s=s)
    cs_z = UnivariateSpline(t, points[:, 2], s=s)

    return cs_x, cs_y, cs_z, t


def plot_3d_curve(ax, curve):
    cs_x, cs_y, cs_z, t = curve
    t_new = np.linspace(0, t[-1], 84)
    curve_points_x = cs_x(t_new)
    curve_points_y = cs_y(t_new)
    curve_points_z = cs_z(t_new)

    ax.plot3D(curve_points_x, curve_points_y, curve_points_z, "r-", linewidth=1)


def get_distance_to_curve3d(points_3D, curve):
    cs_x, cs_y, cs_z, t = curve
    t_new = np.linspace(0, t[-1], 1000)  # Adjust the number of points as needed
    curve_points = np.vstack((cs_x(t_new), cs_y(t_new), cs_z(t_new))).T

    distances = []
    for point_3D in points_3D:
        distance_array = np.linalg.norm(curve_points - point_3D, axis=1)
        distances.append(np.min(distance_array))
    return distances


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


def bound_points(points):
    points = points[points[:, 0] < top.shape[1]]
    points = points[points[:, 1] < top.shape[0]]
    points = points[points[:, 0] > 0]
    points = points[points[:, 1] > 0]
    return points


def visualize_points_2d(images: list, point_sets: list):
    images = [image.copy() for image in images]
    for i in range(len(images)):
        for point in point_sets[i]:
            cv2.circle(images[i], tuple(point.astype(int)), 1, (0, 255, 0), -1)
    combined = np.concatenate(images, axis=1)
    cv2.imshow("combined", combined)
    cv2.waitKey(1)


def visualize_2d(
    images: tuple[np.ndarray], point_sets: tuple[list[np.ndarray]], labels: list
):
    fig, ax = plt.subplots(1, len(images))
    for i, (image, points, labels) in enumerate(zip(images, point_sets, labels)):
        for p in points:
            ax[i].plot(p[:, 0], p[:, 1], ".", label=labels)
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.1, 1.1))
    plt.show()


def visualize_3d(
    point_sets: list,
    labels: list,
    colors: list = ["r", "g"],
):
    fig = plt.figure(figsize=(4, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    for points, label in zip(point_sets, labels):
        if label == "Actual":
            continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                # ".",
                label=label,
                color=colors.pop(0),
            )
        else:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                ".",
                label=label,
                s=5,
                color=colors.pop(0),
            )
            # make a lineplot of the curve
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="r")

    set_limits(ax)

    ax.view_init(elev=0, azim=180)
    # make tick labels invisible
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    # fig.legend(
    #     ncol=1,
    #     loc="outside left center",
    #     markerscale=5,
    #     # text size
    #     prop={"size": 8},
    #     # bbox_to_anchor=(
    #     #     -0.0,
    #     #     0.5,
    #     # ),
    # )
    fig.savefig(
        "./data/figures/3d.png",
        dpi=300 * 4,
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()


def visualize_curves_2d(images: list, curves: list):
    fig, ax = plt.subplots(1, len(images))
    for i, (image, curve) in enumerate(zip(images, curves)):
        plot_curve_on_image(ax[i], curve)
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.1, 1.1))
    plt.show()


def distances_to_curve(point_sets: list, curve: list, labels: list):
    string = ""
    for points, curve, label in zip(point_sets, curve, labels):
        if points.shape[1] == 2:
            distances = get_distance_to_curve(points, curve)
        else:
            distances = get_distance_to_curve3d(points, curve)
        string += f"{label}: \n\
        \tMean: {np.mean(distances).round(4)} \n\
        \tMedian: {np.median(distances).round(4)} \n\
        \tMax: {np.max(distances).round(4)} \n\
        \tMin: {np.min(distances).round(4)} \n"
    return string


def curve_constraint(points, curve3d):
    points = points.reshape(-1, 3)
    return np.sum(get_distance_to_curve3d(points, curve3d))


def objective_function(
    params: np.ndarray,
    curve1: np.ndarray,
    curve2: np.ndarray,
    curve3d: np.ndarray,
) -> float:
    def fn_loss(points, line):
        distance_to_curve = get_distance_to_curve(points, line)
        return distance_to_curve

    points_3d = params.reshape(-1, 3)

    projected_points1 = point2pixel(points_3d, P_TOP)
    projected_points2 = point2pixel(points_3d, P_SIDE)

    error1 = fn_loss(projected_points1, curve1)
    error2 = fn_loss(projected_points2, curve2)

    total_error = np.sum(error1) + np.sum(error2)

    return total_error


def optimize_projection(
    points_3d: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    def callback(intermediate_result):
        global iteration

        params = intermediate_result.x  # extract current parameters from OptimizeResult
        points_3d = params.reshape(-1, 3)
        projected_points1 = point2pixel(points_3d, P_TOP)
        projected_points2 = point2pixel(points_3d, P_SIDE)

        visualize_points_2d([top, side], [projected_points1, projected_points2])

        error1 = get_distance_to_curve(projected_points1, curve1)
        error2 = get_distance_to_curve(projected_points2, curve2)
        total_error = np.sum(error1) + np.sum(error2)

        d = distances_to_curve(
            [projected_points1, projected_points2, points_3d],
            [curve1, curve2, curve3d],
            ["Top", "Side", "3D"],
        )
        print(f"Iteration {iteration}: {total_error}\n{d}", end="\r")
        iteration += 1

    def fn_constraint(points):
        def equidistance_constraint(points):
            points = points.reshape(-1, 3)
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            return np.sum(np.abs(distances - 0.002))

        return equidistance_constraint(points)

    def tip_constraint(points):
        points = points.reshape(-1, 3)
        return np.linalg.norm(points[0] - points_3d[0])

    curve1 = get_curve_2d(points1)
    curve2 = get_curve_2d(points2)
    curve3d = get_curve_3D(points_3d)

    def dist_constraint(points):
        points = points.reshape(-1, 3)
        distances = np.array(get_distance_to_curve3d(points, curve3d))
        return np.sum(np.maximum(0.1 - distances, 0))

    resampled_points = sample_points(curve3d)

    params_init = resampled_points.flatten()

    params_opt = optimize.minimize(
        objective_function,
        params_init,
        args=(curve1, curve2, curve3d),
        method="trust-constr",
        options={
            "maxiter": 100,
            "disp": True,
        },
        constraints=(
            {"type": "eq", "fun": fn_constraint},
            {"type": "eq", "fun": tip_constraint},
            # {"type": "ineq", "fun": dist_constraint},
        ),
        callback=callback,
    )

    print(f"Optimization result: {params_opt}")

    points_3d_opt = params_opt.x.reshape(-1, 3)

    return points_3d_opt


def visualize_points_2d_matplotlib(
    images: list, point_sets: list, labels: list, titles: list
):
    fig, ax = plt.subplots(1, len(images))
    images = [image.copy() for image in images]
    for i in range(len(images)):
        for j, point in enumerate(point_sets[i]):
            ax[i].plot(point[:, 0], point[:, 1], ".", label=labels[j])
        ax[i].imshow(images[i])
        ax[i].axis("off")
        ax[i].set_title(titles[i])
    handles, labels = ax[0].get_legend_handles_labels()
    # reduce the space between legend and the figure
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        # loc="upper center",
        markerscale=5,
        # frameon=True,
        # bbox_to_anchor=(0.26, 1, 1, 0),
        loc="outside upper center",
        # mode="expand",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    # fig.savefig("./data/figures/reprojection.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    top_actual = point2pixel(geom_pos, P_TOP)
    top_actual = bound_points(top_actual)
    side_actual = point2pixel(geom_pos, P_SIDE)
    side_actual = bound_points(side_actual)

    curve_3d = get_curve_3D(points_3d)
    generated = sample_points(curve_3d, distance=0.014)
    top_reprojected = point2pixel(generated, P_TOP)
    side_reprojected = point2pixel(generated, P_SIDE)

    # visualize_points_2d_matplotlib(
    #     [top, side],
    #     ([top_actual, top_reprojected], [side_actual, side_reprojected]),
    #     ["Actual", "Reprojected"],
    #     ["Top", "Side"],
    # )

    # visualize_2d(
    #     [top, side],
    #     ([top_actual, side_actual], [top_reprojected, side_reprojected]),
    #     ["Original", "Reprojected"],
    # )

    visualize_3d([geom_pos, generated], ["Actual", "Generated"])
    exit()

    top_curve = get_curve_2d(top_actual)
    top_distances = get_distance_to_curve(top_actual, top_curve)
    top_distances_reproj = get_distance_to_curve(top_reprojected, top_curve)

    side_curve = get_curve_2d(side_actual)
    side_distances = get_distance_to_curve(side_actual, side_curve)
    side_distances_reproj = get_distance_to_curve(side_reprojected, side_curve)

    visualize_curves_2d([top, side], [top_curve, side_curve])
    # exit()
    print(
        f"Top view: \n\tOriginal: {np.mean(top_distances)} \
        \tReprojected: {np.mean(top_distances_reproj)}"
    )
    print(
        f"Side view: \n\tOriginal: {np.mean(side_distances)} \
        \tReprojected: {np.mean(side_distances_reproj)}"
    )
    print(
        f"3D curve distances: {np.mean(get_distance_to_curve3d(points_3d, curve_3d))}"
    )
    distances_to_curve(
        [top_actual, side_actual, points_3d],
        [top_curve, side_curve, curve_3d],
        ["Top", "Side", "3D"],
    )

    # distances_original = get_distance_to_curve(points1, curve)
    # distances_reproj = get_distance_to_curve(reprojected, curve)
    reproj_opt = optimize_projection(points_3d, points1, points2)
    print(
        "Mean offset before:",
        np.mean(points_3d - geom_pos[: len(points_3d)]),
    )
    print(
        "Mean offset after:",
        np.mean(reproj_opt - geom_pos[: len(reproj_opt)]),
    )
    reproj_opt_top = point2pixel(reproj_opt, P_TOP)
    reproj_opt_side = point2pixel(reproj_opt, P_SIDE)

    visualize_2d(
        [top, side],
        ([top_actual, side_actual], [reproj_opt_top, reproj_opt_side]),
        ["Actual", "Reprojected"],
    )

    visualize_3d([geom_pos, reproj_opt], ["Original", "Reprojected"])
