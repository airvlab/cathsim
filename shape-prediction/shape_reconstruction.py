from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from cathsim.dm.visualization import point2pixel
from scipy import interpolate, optimize
from skimage.morphology import thin

# set scatter marker size
mpl.rcParams["lines.markersize"] = 1

DATA_DIR = Path.cwd() / "data" / "guidewire-reconstruction"

# if not DATA_DIR.exists():
#     raise FileNotFoundError(f"{DATA_DIR} does not exist.")

P_TOP = np.array(
    [
        [-5.79411255e02, 0.00000000e00, 2.39500000e02, -7.72573376e01],
        [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.32301407e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00, -2.50000000e-01],
    ]
)

P_SIDE = np.array(
    [
        [-2.39500000e02, 5.79411255e02, 0.00000000e00, -1.13528182e02],
        [-2.39500000e02, 0.00000000e00, 5.79411255e02, -7.00723376e01],
        [-1.00000000e00, 0.00000000e00, 0.00000000e00, -2.20000000e-01],
    ]
)


def set_limits(ax, plot_mesh=False):
    mesh_path = (
        Path.cwd()
        / "src/cathsim/dm/components/phantom_assets/meshes/phantom3/visual.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    if plot_mesh:
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            alpha=0.1,
            color="gray",
        )
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])


def read_segmented_image(file_path):
    return cv2.imread(file_path.as_posix(), 0)


def find_endpoints(skeleton):
    endpoints = []
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j]:
                sum_neighbors = np.sum(skeleton[i - 1 : i + 2, j - 1 : j + 2])
                if sum_neighbors == 2:  # The pixel itself + one neighbor = 2
                    # print(f"Found endpoint at ({j}, {i})")
                    endpoints.append((j, i))
    return endpoints


def get_backbone_points(image, spacing=1):
    skeleton = thin(image)

    def neighbors(x, y, shape):
        for i in range(max(0, x - 1), min(shape[0], x + 2)):
            for j in range(max(0, y - 1), min(shape[1], y + 2)):
                yield i, j

    endpoint = find_endpoints(skeleton)[0]

    visited = set()
    path = []
    to_explore = [endpoint]

    while to_explore:
        x, y = to_explore.pop()
        if (x, y) in visited:
            continue

        visited.add((x, y))
        path.append((x, y))

        for i, j in neighbors(x, y, skeleton.shape):
            if skeleton[j, i] and (i, j) not in visited:
                to_explore.append((i, j))

    path = path[::spacing]
    return np.array(path)


def compute_gradient(P_3d, P_2d_observed, P_matrix, alpha=1.0):
    homogeneous_P_3d = np.hstack(
        [P_3d, np.ones((P_3d.shape[0], 1))]
    )  # Convert to homogeneous coordinates
    P_2d_reprojected = np.dot(P_matrix, homogeneous_P_3d.T).T
    P_2d_reprojected /= P_2d_reprojected[:, 2][:, np.newaxis]

    error = P_2d_reprojected[:, :2] - P_2d_observed  # shape (n, 2)

    J = P_matrix[:2, :3]  # The Jacobian matrix with respect to x, y, and z

    gradient = np.dot(error, J) * alpha  # shape should be (n, 3)

    P_3d_new = P_3d - gradient  # Update 3D points based on gradient

    return P_3d_new


def reprojection_error_and_gradient(
    points_3d, points2d_1, points2d_2, P1, P2, alpha=1.0
):
    error_1 = np.sum((points2d_1 - point2pixel(points_3d, P1)) ** 2)
    error_2 = np.sum((points2d_2 - point2pixel(points_3d, P2)) ** 2)
    total_error = error_1 + error_2

    gradient_1 = compute_gradient(points_3d, points2d_1, P1, alpha=alpha)
    gradient_2 = compute_gradient(points_3d, points2d_2, P2, alpha=alpha)

    gradient_combined = (gradient_1 + gradient_2) / 2.0  # Averaging the gradients

    return total_error, gradient_combined


def plot_over_image(image, points, alternate_color=False, size=3):
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    image = image.copy()
    image = np.stack((image,) * 3, axis=-1)
    for i, p in enumerate(points):
        color = colors[i % 3] if alternate_color else (0, 0, 255)
        cv2.circle(image, tuple(p), 0, color, size)
    return image


def valid_points(points: np.ndarray):
    if np.any(points[3, :] == 0) or np.any(np.isnan(points[3, :])):
        # get the mask and remove it from points1 and points2 and points_3d_h
        mask = np.logical_and(points[3, :] != 0, ~np.isnan(points[3, :]))
        print(f"Foudn {np.sum(~mask)} invalid points")
        return mask


def triangulate_points(P1, P2, x1s, x2s):
    """
    P1, P2: Projection matrices for camera 1 and 2, shape 3x4
    x1s, x2s: Homogeneous coordinates of the points in image 1 and image 2, shape Nx3

    Returns:
    Xs: Homogeneous coordinates of the 3D points, shape Nx4
    """
    N = x1s.shape[0]
    Xs = np.zeros((N, 4))

    for i in range(N):
        x1 = x1s[i]
        x2 = x2s[i]

        A = np.zeros((4, 4))
        A[0:2, :] = x1[0] * P1[2, :] - P1[0, :]
        A[2:4, :] = x1[1] * P1[2, :] - P1[1, :]

        B = np.zeros((4, 4))
        B[0:2, :] = x2[0] * P2[2, :] - P2[0, :]
        B[2:4, :] = x2[1] * P2[2, :] - P2[1, :]

        # Stacking A and B to form a 4x4 matrix
        A = np.vstack((A, B))

        # Solve for X by minimizing ||AX|| subject to ||X|| = 1
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        Xs[i, :] = X / X[-1]  # Dehomogenize (make last element 1)

    return Xs


def reprojection_error(
    params: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    P_top: np.ndarray,
    P_side: np.ndarray,
    fn_loss: callable,
    fn_aggregate: callable = None,
) -> np.ndarray:
    # Reshape params into 3D points
    points_3d = params.reshape(-1, 3)

    # Project the 3D points back to 2D points using projection matrices
    projected_points1 = point2pixel(points_3d, P_top)
    projected_points2 = point2pixel(points_3d, P_side)

    assert (
        projected_points1.shape == points1.shape
        and projected_points2.shape == points2.shape
    ), (
        f"projected_points1.shape = {projected_points1.shape}, points1.shape = {points1.shape}, "
        f"projected_points2.shape = {projected_points2.shape}, points2.shape = {points2.shape}"
    )

    # Compute the reprojection error
    error1 = fn_loss(points1, projected_points1)
    error2 = fn_loss(points2, projected_points2)

    if fn_aggregate is None:
        return error1 + error2
    total_error = fn_aggregate(error1, error2)
    print(f"Total error: {total_error}")

    return total_error


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=-1)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.sum(0.5 * quadratic**2 + delta * linear, axis=-1)


def aggregate_error(error1, error2):
    return np.sum(error1) + np.sum(error2)


def get_points(
    image_num: int = 23, spacing: int = 15, size: int = 10, debug: bool = False
):
    img1 = read_segmented_image(DATA_DIR / f"{image_num}_top.png")
    img2 = read_segmented_image(DATA_DIR / f"{image_num}_side.png")
    actual = np.load(DATA_DIR / f"{image_num}_actual.npy")
    # print(actual.shape)

    points1 = get_backbone_points(img1, spacing=spacing)
    points2 = get_backbone_points(img2, spacing=spacing)
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

    if debug:
        print(points_3d_h.shape)
        img1_w_points = plot_over_image(img1, points1, alternate_color=True, size=size)
        img2_w_points = plot_over_image(img2, points2, alternate_color=True, size=size)
        combined = np.hstack([img1_w_points, img2_w_points])
        plt.imshow(combined)
        plt.axis("off")
    assert (
        points1.shape == points2.shape and points2.shape[0] == points_3d.shape[0]
    ), f"points1.shape = {points1.shape}, points2.shape = {points2.shape}, points_3d.shape = {points_3d.shape}"
    return points1, points2, points_3d, actual


def optimize_projection(
    points_3d: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    P1: np.ndarray = P_TOP,
    P2: np.ndarray = P_SIDE,
) -> np.ndarray:
    # Flatten the 3D points as initial parameters
    params_init = points_3d.flatten()
    # optimize the 3D points
    params_opt = optimize.minimize(
        reprojection_error,
        params_init,
        args=(points1, points2, P_TOP, P_SIDE, manhattan_distance, aggregate_error),
        method="CG",
        options={
            "maxiter": 1e7,
            "disp": True,
            "eps": 1e-6,  # Step size used for numerical approximation of the Jacobian
            "gtol": 1e-6,  # Tolerance for termination by the change of the objective function value
        },
    )

    print(f"Optimization result: {params_opt}")
    points_3d_opt = params_opt.x.reshape(-1, 3)

    return points_3d_opt


def fit_3d_curve(points_3d):
    rbfi = interpolate.Rbf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    return rbfi


def point_on_curve(curve_function, t):
    # Get a 3D point corresponding to parameter t on the curve
    return curve_function(t)


def reconstruct_3d_points(params):
    pass


def measure_distance_to_backbone(projected_points, backbone):
    pass


def spacing_constraint(points_3d, target_spacing=0.002):
    distances = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
    return np.sum(np.abs(distances - target_spacing))


def objective_function(
    params: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    fn_loss: callable,
    fn_aggregate: callable = None,
    fn_constraint: callable = None,
) -> float:
    points_3d = params.reshape(-1, 3)
    projected_points1 = point2pixel(points_3d, P1)
    projected_points2 = point2pixel(points_3d, P2)

    error1 = fn_loss(points1, projected_points1)
    error2 = fn_loss(points2, projected_points2)

    if fn_aggregate is None:
        total_error = error1 + error2
    else:
        total_error = fn_aggregate(error1, error2)

    if fn_constraint is not None:
        constraint_error = fn_constraint(points_3d)
        total_error += constraint_error

    return total_error


def optimize_projection_2(
    points_3d: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    P1: np.ndarray = P_TOP,
    P2: np.ndarray = P_SIDE,
    fn_loss: callable = manhattan_distance,
    fn_aggregate: callable = aggregate_error,
    fn_constraint: callable = spacing_constraint,
) -> np.ndarray:
    params_init = points_3d.flatten()

    params_opt = optimize.minimize(
        objective_function,
        params_init,
        args=(points1, points2, P1, P2, fn_loss, fn_aggregate, fn_constraint),
        method="CG",
        options={
            "maxiter": 1e7,
            "disp": True,
            "eps": 1e-6,
            "gtol": 1e-6,
        },
    )

    print(f"Optimization result: {params_opt}")
    points_3d_opt = params_opt.x.reshape(-1, 3)

    return points_3d_opt


def make_3d_curve(points_3d):
    # Compute pairwise distances and total length of the piecewise linear path
    dists = np.sqrt(np.sum(np.diff(points_3d, axis=0) ** 2, axis=1))
    total_length = np.sum(dists)

    # Compute the arc-length parameterization
    u = np.zeros_like(dists)
    u[1:] = np.cumsum(dists[:-1]) / total_length

    # Fit polynomial curves in x, y, z separately
    degree = 5  # for example
    p_x = np.polyfit(u, points_3d[:-1, 0], degree)
    p_y = np.polyfit(u, points_3d[:-1, 1], degree)
    p_z = np.polyfit(u, points_3d[:-1, 2], degree)

    # Resample at uniform intervals along the curve
    num_points = int(total_length / 0.002)  # 2mm spacing
    u_new = np.linspace(0, 1, num_points)
    x_new = np.polyval(p_x, u_new)
    y_new = np.polyval(p_y, u_new)
    z_new = np.polyval(p_z, u_new)
    points_3d_new = np.column_stack((x_new, y_new, z_new))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c="b",
        label="Original points",
        s=1,
    )
    ax.plot(
        points_3d_new[:, 0],
        points_3d_new[:, 1],
        points_3d_new[:, 2],
        c="r",
        label="Fitted curve",
    )
    set_limits(ax)

    return fig, ax


def interpolate_even_spacing(chain, spacing=0.002):
    new_chain = [chain[0]]
    leftover_distance = 0

    for i in range(len(chain) - 1):
        start = chain[i]
        end = chain[i + 1]

        segment = end - start
        segment_length = np.linalg.norm(segment)

        num_points = int((segment_length + leftover_distance) // spacing)
        leftover_distance += segment_length - num_points * spacing

        for j in range(1, num_points + 1):
            fraction = (j * spacing - leftover_distance) / segment_length
            new_point = start + fraction * segment
            new_chain.append(new_point)

    return np.array(new_chain)


def make_data():
    from cathsim.dm import make_dm_env
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2
    import random

    data_path = Path.cwd() / "data"

    def save_guidewire_reconstruction(
        t: int,
        top: np.ndarray,
        side: np.ndarray,
        geom_pos: np.ndarray,
        path: Path = data_path / "guidewire-reconstruction",
    ):
        if not path.exists():
            path.mkdir(parents=True)

        plt.imsave(path / f"{t}_top.png", top[:, :, 0], cmap="gray")
        plt.imsave(path / f"{t}_side.png", side[:, :, 0], cmap="gray")
        np.save(path / f"{t}_actual", geom_pos)

    env = make_dm_env(
        phantom="phantom3",
        use_pixels=True,
        use_segment=True,
        use_side=True,
        image_size=480,
        target_from_sites=False,
        # apply_fluid_force=True,
    )

    def policy(time_step):
        del time_step
        return [0.4, random.random() * 2 - 1]

    for episode in range(1):
        time_step = env.reset()
        for step in range(100):
            action = policy(time_step)
            top = time_step.observation["guidewire"]
            side = time_step.observation["side"]
            geom_pos = env.task.get_geom_pos(env.physics)
            save_guidewire_reconstruction(step, top, side, geom_pos)
            cv2.imshow("top", top)
            cv2.waitKey(1)
            time_step = env.step(action)


if __name__ == "__main__":
    # make_data()
    # exit()
    points1, points2, points_3d, geom_pos = get_points()
    image = read_segmented_image(DATA_DIR / "23_side.png")
    # points_3d_opt = optimize_projection_2(points_3d, points1, points2)
    fig, ax = make_3d_curve(points_3d)
    # equidistant_points = interpolate_even_spacing(points_3d)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    set_limits(ax)
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', label='Original points', s=1)
    # ax.scatter(equidistant_points[:, 0], equidistant_points[:, 1], equidistant_points[:, 2], c='b', label='Equidistant points', s=1)
    ax.scatter(
        geom_pos[:, 0], geom_pos[:, 1], geom_pos[:, 2], c="g", label="Geom points", s=1
    )
    ax.legend(ncol=3)
    plt.show()

    # points_3d_opt = optimize_projection(points_3d, points1, points2)
    # side_actual = plot_over_image(image, point2pixel(geom_pos, P_SIDE), alternate_color=True, size=3)
    # side_points = plot_over_image(image, points2, alternate_color=True, size=3)
    # side_reprojected_opt = plot_over_image(image, point2pixel(points_3d_opt, P_SIDE), alternate_color=True, size=3)
    # combined = np.hstack([side_actual, side_points, side_reprojected_opt])
    # plt.imshow(combined)
    # plt.axis("off")
    # plt.show()
