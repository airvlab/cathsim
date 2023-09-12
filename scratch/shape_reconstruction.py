import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cathsim.dm.visualization import point2pixel

from guidewire_reconstruction_3 import (
    get_backbone_points,
    P_top,
    P_side,
    DATA_DIR,
    read_segmented_image,
    plot3d,
    plot_over_image,
)


def draw_epilines(img: np.ndarray, lines: np.ndarray):
    img = img.copy()
    r, c = img.shape
    for r, pt1, pt2 in zip(lines, points1, points2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv2.circle(img, tuple(pt1), 5, color, -1)
        img = cv2.circle(img, tuple(pt2), 5, color, -1)
    return img


def reproject_points(P_3d, P_matrix):
    # Augment the 3D points to homogeneous coordinates
    P_3d_h = np.hstack((P_3d, np.ones((P_3d.shape[0], 1))))

    # Reproject
    P_2d_h = P_3d_h.dot(P_matrix.T)

    # Convert to Cartesian coordinates
    P_2d = P_2d_h[:, :2] / P_2d_h[:, 2:]

    return P_2d


def compute_gradient(P_3d, P_2d_observed, P_matrix, alpha=1.0):
    P_2d_reprojected = reproject_points(P_3d, P_matrix)
    error = P_2d_reprojected - P_2d_observed  # shape (n, 2)

    # Jacobian with respect to the homogeneous coordinate system
    J_h = np.zeros((P_3d.shape[0], 3, 3))  # shape (n, 3, 3)

    J_h[:, 0, 0] = P_matrix[0, 0]
    J_h[:, 0, 1] = P_matrix[0, 1]
    J_h[:, 0, 2] = P_matrix[0, 2]

    J_h[:, 1, 0] = P_matrix[1, 0]
    J_h[:, 1, 1] = P_matrix[1, 1]
    J_h[:, 1, 2] = P_matrix[1, 2]

    # Update 3D points
    gradient = np.zeros((P_3d.shape[0], 3))  # shape (n, 3)
    for i in range(P_3d.shape[0]):
        for d in range(2):  # for each dimension in 2D
            gradient[i, :] += alpha * error[i, d] * J_h[i, d, :]

    P_3d_new = P_3d - gradient
    return P_3d_new


def reprojection_error_and_gradient(points_3d, points2d_1, points2d_2, P1, P2, alpha=1.0):
    # Calculate reprojection error for both sets of 2D points
    points2d_1_reprojected = reproject_points(points_3d, P1)
    points2d_2_reprojected = reproject_points(points_3d, P2)
    error_1 = np.sum((points2d_1 - points2d_1_reprojected) ** 2)
    error_2 = np.sum((points2d_2 - points2d_2_reprojected) ** 2)
    total_error = error_1 + error_2

    # Calculate gradient for both sets of 2D points
    gradient_1 = compute_gradient(points_3d, points2d_1, P1, alpha=alpha)
    gradient_2 = compute_gradient(points_3d, points2d_2, P2, alpha=alpha)

    # Combine the gradients (here I'm simply averaging them)
    gradient_combined = (gradient_1 + gradient_2) / 2.0

    return total_error, gradient_combined


if __name__ == "__main__":

    head_pos = DATA_DIR / "14_geom_pos.npy"
    head_pos = np.load(head_pos)

    img1 = read_segmented_image(DATA_DIR / "14_top.png")
    img2 = read_segmented_image(DATA_DIR / "14_side.png")
    points1, _ = get_backbone_points(img1)
    points2, _ = get_backbone_points(img2)
    min_length = min(len(points1), len(points2))
    # points1 = points1[:min_length]
    # points2 = points2[:min_length]

    img1_w_points = plot_over_image(img1, points1, alternate_color=True)
    img2_w_points = plot_over_image(img2, points2, alternate_color=True)
    combined = np.hstack([img1_w_points, img2_w_points])

    plt.imshow(combined)
    plt.axis("off")
    plt.show()
    exit()

    # switch x and y to image coordinates
    # points1 = np.flip(points1, axis=1)
    # points2 = np.flip(points2, axis=1)

    points_3d_h = cv2.triangulatePoints(P_top, P_side, points1.T, points2.T)
    print(points_3d_h.shape)
    if np.any(points_3d_h[3, :] == 0) or np.any(np.isnan(points_3d_h[3, :])):
        # get the mask and remove it from points1 and points2 and points_3d_h
        mask = np.logical_and(points_3d_h[3, :] != 0, ~np.isnan(points_3d_h[3, :]))
        points1 = points1[mask]
        points2 = points2[mask]
        points_3d_h = points_3d_h[:, mask]

    points_3d = points_3d_h[:3, :] / points_3d_h[3, :]
    points_3d = points_3d.T

    alpha = 0.1  # Learning rate, you may need to adjust this
    num_iterations = 100  # Number of iterations

    for i in range(num_iterations):
        total_error, gradient = reprojection_error_and_gradient(points_3d, points1, points2, P_top, P_side, alpha=alpha)
        print(f"Iteration {i+1}: Total Error = {total_error}",)
        points_3d -= alpha * gradient
