import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cathsim.dm.visualization import point2pixel

from guidewire_reconstruction import (
    get_backbone_points,
    P_top,
    P_side,
    DATA_DIR,
    read_segmented_image,
    plot3d,
)


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


def plot_over_image(image, points, alternate_color=False):
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    image = image.copy()
    image = np.stack((image,) * 3, axis=-1)
    for i, p in enumerate(points):
        color = colors[i % 3] if alternate_color else (0, 0, 255)
        cv2.circle(image, tuple(p), 0, color, 3)
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

    img1 = read_segmented_image(DATA_DIR / f"{image_num}_top.png")
    img2 = read_segmented_image(DATA_DIR / f"{image_num}_side.png")
    points1 = get_backbone_points(img1, spacing=4)
    points2 = get_backbone_points(img2, spacing=4)
    min_length = min(len(points1), len(points2))
    points1 = points1[:min_length]
    points2 = points2[:min_length]
    print(f"Number of points: {len(points1)}")

    img1_w_points = plot_over_image(img1, points1, alternate_color=True)
    img2_w_points = plot_over_image(img2, points2, alternate_color=True)
    combined = np.hstack([img1_w_points, img2_w_points])

    # plt.imshow(combined)
    # plt.axis("off")
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

    img1_w_points = plot_over_image(img1, reprojected_top, alternate_color=True)
    img2_w_points = plot_over_image(img2, reprojected_side, alternate_color=True)
    combined = np.hstack([img1_w_points, img2_w_points])
    plt.imshow(combined)
    plt.axis("off")

    __import__('pprint').pprint(points_3d)
    # plot3d(points_3d, head_pos)

    alpha = 0.003  # Learning rate, you may need to adjust this
    num_iterations = 10  # Number of iterations
    for i in range(num_iterations):
        total_error, gradient = reprojection_error_and_gradient(points_3d, points1, points2, P_top, P_side, alpha=alpha)
        print(f"Iteration {i+1}: Total Error = {total_error}",)
        points_3d -= alpha * gradient
    # plot3d(points_3d, head_pos)
