import cv2
import numpy as np
from pathlib import Path
from cathsim.dm.visualization import point2pixel, plot_w_mesh
import matplotlib.pyplot as plt
import trimesh
from pprint import pformat

DATA_DIR = Path.cwd() / "data" / "guidewire-reconstruction-2"

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


if not DATA_DIR.exists():
    raise FileNotFoundError(f"{DATA_DIR} does not exist.")


def read_segmented_image(file_path):
    return cv2.imread(file_path.as_posix(), cv2.IMREAD_GRAYSCALE)


def plot_point_on_image(image, point, camera_matrix=None, color=(0, 0, 255), radius=5):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if camera_matrix is not None:
        point = point2pixel(point, camera_matrix)
    point = tuple(point.astype(int))

    cv2.circle(image, point, radius, color, -1)

    return image


def plot_two(top, side):
    combined = np.hstack([top, side])
    plt.imshow(combined)
    plt.axis("off")
    plt.show()


def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    # Get a cross-shaped structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Step 1: Open the image
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        # Step 2: Substract open from the original image
        temp = cv2.subtract(image, opened)
        # Step 3: Erode the original image and refine the skeleton
        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        # Step 4: If there are no white pixels left, exit the loop
        if cv2.countNonZero(image) == 0:
            break

    return skel


def find_skeleton_points(image, debug=False, sampling_interval=10):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Skeletonize the image
    skel = skeletonize(gray)  # assuming skeletonize() is your skeletonization function

    if debug:
        print("Skeleton found")

    # Find coordinates of all non-zero pixels in the skeleton
    coords = np.column_stack(np.where(skel > 0))

    # Sample points along the skeleton at regular intervals
    sampled_coords = coords[::sampling_interval]

    if debug:
        print(f"Sampled coordinates: {sampled_coords}")

    return np.array(sampled_coords)


def plot_centroids(image, centroids):
    # switch x and y
    centroids = centroids[:, ::-1]
    for centroid in centroids:
        image = plot_point_on_image(image, centroid, color=(255, 0, 0), radius=2)
    return image


def triangulate_points(projMat1, projMat2, points1, points2):
    length = min(len(points1), len(points2))
    points1 = points1[:length]
    points2 = points2[:length]

    assert (points1.shape == points2.shape)

    points_3d_h = cv2.triangulatePoints(projMat1, projMat2, points1.T, points2.T)
    points_3d = points_3d_h[:3, :] / points_3d_h[3, :]

    assert (points_3d.shape == (3, length))

    mask = np.logical_and(points_3d_h[3, :] != 0, np.isfinite(points_3d_h[3, :]))
    filtered_points_3d_h = points_3d_h[:, mask]
    points_3d = filtered_points_3d_h[:3, :] / filtered_points_3d_h[3, :]
    print("Points 3D: \n", pformat(points_3d.T))

    return points_3d.T


def plot3d(rec_points, head_pos):
    mesh_path = Path.cwd() / "src/cathsim/dm/components/phantom_assets/meshes/phantom3/visual.stl"
    mesh = trimesh.load_mesh(mesh_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], label="Reconstructed points", s=2)
    ax.scatter(head_positions[:, 0], head_positions[:, 1], head_positions[:, 2], label="Head positions", s=2)
    plt.show()


def find_fundamental_matrix(points1, points2):
    # Using 8-point algorithm
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    return F, mask


def find_corresponding_points(lines, points):
    # For simplicity, let's find the nearest point to the epipolar line.
    # lines are Ax + By + C = 0
    # points are (x, y)
    corresponding_points = []
    for line in lines:
        A, B, C = line
        min_distance = float("inf")
        best_point = None
        for point in points:
            x, y = point
            # Compute the distance from the point to the line
            distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
            if distance < min_distance:
                min_distance = distance
                best_point = point
        corresponding_points.append(best_point)
    return np.array(corresponding_points)

    # Your remaining code goes here
if __name__ == "__main__":
    top_view = read_segmented_image(DATA_DIR / "14_top.png")
    side_view = read_segmented_image(DATA_DIR / "14_side.png")
    head_positions = np.load(DATA_DIR / "14_geom_pos.npy")
    head_pos = head_positions[-1]

    top_skeleton_points = find_skeleton_points(top_view, debug=True)
    side_skeleton_points = find_skeleton_points(side_view, debug=True)
    length = min(len(top_skeleton_points), len(side_skeleton_points))
    top_skeleton_points = top_skeleton_points[:length]
    side_skeleton_points = side_skeleton_points[:length]

   # Find the Fundamental Matrix
    F, mask = find_fundamental_matrix(top_skeleton_points, side_skeleton_points)

    # Compute epipolar lines
    lines_in_side = cv2.computeCorrespondEpilines(top_skeleton_points.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines_in_top = cv2.computeCorrespondEpilines(side_skeleton_points.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    # Find the corresponding points along the epipolar lines
    corr_points_side = find_corresponding_points(lines_in_side, side_skeleton_points)
    corr_points_top = find_corresponding_points(lines_in_top, top_skeleton_points)

    # Perform the triangulation with corresponding points
    points3D = triangulate_points(P_top, P_side, corr_points_top, corr_points_side)
    print("Shape of points3D:", points3D.shape)

    plot3d(points3D, head_pos)

    exit()

    # plot_two(top_view, side_view)
    visualize_top = plot_point_on_image(top_view, head_pos, P_top)
    visualize_top = plot_centroids(top_view, top_skeleton_points)
    visualize_side = plot_point_on_image(side_view, head_pos, P_side)
    visualize_side = plot_centroids(side_view, side_skeleton_points)
    # plot_two(visualize_top, visualize_side)

    points3D = triangulate_points(P_top, P_side, top_skeleton_points, side_skeleton_points)
    print("Shape of points3D:", points3D.shape)

    exit()
    top_view = read_segmented_image(DATA_DIR / "14_top.png")
    side_view = read_segmented_image(DATA_DIR / "14_side.png")

    for point in points3D:
        top_view = plot_point_on_image(top_view, point, P_top, color=(0, 255, 0), radius=2)
        side_view = plot_point_on_image(side_view, point, P_side, color=(0, 255, 0), radius=2)
