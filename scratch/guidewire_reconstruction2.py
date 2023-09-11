import cv2
import numpy as np
from pathlib import Path
from cathsim.dm.visualization import point2pixel
import matplotlib.pyplot as plt

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


def plot_point_on_image(image, point, camera_matrix, color=(0, 0, 255), radius=5):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    point = point2pixel(point, camera_matrix)
    # point[0] = image.shape[1] - point[0]
    print(point)
    point = tuple(point.astype(int))

    cv2.circle(image, point, radius, color, -1)

    return image


def read_segmented_image(file_path):
    return cv2.imread(file_path.as_posix(), cv2.IMREAD_GRAYSCALE)


# Dummy function to get camera matrix; replace with actual values
def get_camera_matrix():
    return np.array([[500, 0, 240], [0, 500, 240], [0, 0, 1]])


def find_centroids(image):
    # Your code goes here
    return np.array([[1, 1], [2, 2], [3, 3]])


# Step 1: Read segmented images
top_view = read_segmented_image(DATA_DIR / "14_top.png")
side_view = read_segmented_image(DATA_DIR / "14_side.png")
head_pos = np.load(DATA_DIR / "14_geom_pos.npy")[-1]
print(f"Head position ({head_pos.shape}): {head_pos}")


top_view = plot_point_on_image(top_view, head_pos, P_top)
side_view = plot_point_on_image(side_view, head_pos, P_side)
image = np.hstack([top_view, side_view])

plt.imshow(image)
plt.axis("off")
plt.show()

exit()


# Step 2: Find centroids in each image (you'll need to implement this)
top_centroids = find_centroids(top_view)
side_centroids = find_centroids(side_view)

# Step 3: Camera matrices
camera_matrix_top = get_camera_matrix()
camera_matrix_side = get_camera_matrix()

# Step 4: 3D Reconstruction
# Here, let's assume top_centroids and side_centroids are of shape (N, 2)
# We add a homogeneous coordinate to make them (N, 3)
homogeneous_top_centroids = np.hstack(
    [top_centroids, np.ones((top_centroids.shape[0], 1))]
)
homogeneous_side_centroids = np.hstack(
    [side_centroids, np.ones((side_centroids.shape[0], 1))]
)


# Dummy function to implement triangulation
def triangulate_points(proj_matrix1, proj_matrix2, points1, points2):
    return cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)


# Perform the triangulation
reconstructed_3D_points = triangulate_points(
    camera_matrix_top,
    camera_matrix_side,
    homogeneous_top_centroids,
    homogeneous_side_centroids,
)

# Convert from homogeneous to Euclidean coordinates
reconstructed_3D_points /= reconstructed_3D_points[3, :]
reconstructed_3D_points = reconstructed_3D_points[:3, :]

# Step 5: Optional - Error Minimization

# Step 6: Optional - Validation and Visualization

# Your remaining code goes here
