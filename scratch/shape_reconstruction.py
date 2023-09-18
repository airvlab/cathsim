import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cathsim.dm.visualization import point2pixel
import trimesh
from skimage.morphology import skeletonize, thin

DATA_DIR = Path.cwd() / "data" / "guidewire-reconstruction-2"

if not DATA_DIR.exists():
    raise FileNotFoundError(f"{DATA_DIR} does not exist.")

P_top = np.array([
    [-5.79411255e+02, 0.00000000e+00, 2.39500000e+02, -7.72573376e+01],
    [0.00000000e+00, 5.79411255e+02, 2.39500000e+02, -1.32301407e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -2.50000000e-01],
])

P_side = np.array([
    [-2.39500000e+02, 5.79411255e+02, 0.00000000e+00, -1.13528182e+02],
    [-2.39500000e+02, 0.00000000e+00, 5.79411255e+02, -7.00723376e+01],
    [-1.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.20000000e-01],
])


def plot3d(rec_points=None, head_pos=None):
    mesh_path = Path.cwd() / "src/cathsim/dm/components/phantom_assets/meshes/phantom3/visual.stl"
    mesh = trimesh.load_mesh(mesh_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    if rec_points is not None:
        ax.scatter(rec_points[:, 0], rec_points[:, 1], rec_points[:, 2], label="Reconstructed shape", s=1)
    if head_pos is not None:
        ax.scatter(head_pos[:, 0], head_pos[:, 1], head_pos[:, 2], label="Actual shape", s=1)
    fig.legend(loc="outside upper center", ncol=2)
    plt.show()


def read_segmented_image(file_path):
    return cv2.imread(file_path.as_posix(), 0)


def find_endpoints(skeleton):
    endpoints = []
    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j]:
                sum_neighbors = np.sum(skeleton[i - 1:i + 2, j - 1:j + 2])
                if sum_neighbors == 2:  # The pixel itself + one neighbor = 2
                    print(f"Found endpoint at ({j}, {i})")
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
    homogeneous_P_3d = np.hstack([P_3d, np.ones((P_3d.shape[0], 1))])  # Convert to homogeneous coordinates
    P_2d_reprojected = np.dot(P_matrix, homogeneous_P_3d.T).T
    P_2d_reprojected /= P_2d_reprojected[:, 2][:, np.newaxis]

    error = P_2d_reprojected[:, :2] - P_2d_observed  # shape (n, 2)

    J = P_matrix[:2, :3]  # The Jacobian matrix with respect to x, y, and z

    gradient = np.dot(error, J) * alpha  # shape should be (n, 3)

    P_3d_new = P_3d - gradient  # Update 3D points based on gradient

    return P_3d_new


def reprojection_error_and_gradient(points_3d, points2d_1, points2d_2, P1, P2, alpha=1.0):
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


if __name__ == "__main__":

    image_num = 23
    head_pos = DATA_DIR / f"{image_num}_geom_pos.npy"
    head_pos = np.load(head_pos)
    spacing = 15
    size = 10

    img1 = read_segmented_image(DATA_DIR / f"{image_num}_top.png")
    img2 = read_segmented_image(DATA_DIR / f"{image_num}_side.png")
    points1 = get_backbone_points(img1, spacing=spacing)
    points2 = get_backbone_points(img2, spacing=spacing)
    min_length = min(len(points1), len(points2))
    points1 = points1[:min_length]
    points2 = points2[:min_length]
    print(f"Number of points: {len(points1)}")

    img1_w_points = plot_over_image(img1, points1, alternate_color=True, size=size)
    img2_w_points = plot_over_image(img2, points2, alternate_color=True, size=size)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img1_w_points)
    # ax[1].imshow(img2_w_points)
    # ax[0].title.set_text("Top")
    # ax[1].title.set_text("Side")
    # ax[0].axis("off")
    # ax[1].axis("off")
    # plt.show()

    # points_3d_h = cv2.triangulatePoints(P_top, P_side, points1.T, points2.T)
    points_3d_h = triangulate_points(P_top, P_side, points1, points2)
    print(points_3d_h.shape)
    mask = valid_points(points_3d_h)
    if mask is not None:
        points1 = points1[mask]
        points2 = points2[mask]
        points_3d_h = points_3d_h[mask]

    print(points_3d_h.shape)

    points_3d = points_3d_h[:, :3] / points_3d_h[:, 3, np.newaxis]

    reprojected_top = point2pixel(points_3d, P_top)
    reprojected_side = point2pixel(points_3d, P_side)

    img1_w_points = plot_over_image(img1, reprojected_top, alternate_color=True, size=size)
    img2_w_points = plot_over_image(img2, reprojected_side, alternate_color=True, size=size)
    combined = np.hstack([img1_w_points, img2_w_points])
    plt.imshow(combined)
    plt.axis("off")

    __import__('pprint').pprint(points_3d)
    plot3d(points_3d, head_pos)

    alpha = 0.003  # Learning rate, you may need to adjust this
    num_iterations = 10  # Number of iterations
    for i in range(num_iterations):
        total_error, gradient = reprojection_error_and_gradient(points_3d, points1, points2, P_top, P_side, alpha=alpha)
        print(f"Iteration {i+1}: Total Error = {total_error}",)
        points_3d -= alpha * gradient
    # plot3d(points_3d, head_pos)
