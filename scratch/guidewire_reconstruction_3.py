import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cathsim.dm.visualization import point2pixel
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


def plot_over_image(img, points, alternate_color=False):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color = colors[0]
    i = 0
    for point in points:
        if alternate_color:
            color = colors[i % len(colors)]
            i += 1
        cv2.circle(img, point, 2, color, -1)
    return img


def plot3d(rec_points=None, head_pos=None):
    mesh_path = Path.cwd() / "src/cathsim/dm/components/phantom_assets/meshes/phantom3/visual.stl"
    mesh = trimesh.load_mesh(mesh_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    if rec_points is not None:
        ax.scatter(rec_points[:, 0], rec_points[:, 1], rec_points[:, 2], label="Reconstructed points", s=2)
    if head_pos is not None:
        ax.scatter(head_pos[:, 0], head_pos[:, 1], head_pos[:, 2], label="Head position", s=2)
    plt.show()


if not DATA_DIR.exists():
    raise FileNotFoundError(f"{DATA_DIR} does not exist.")


def read_segmented_image(file_path):
    return cv2.imread(file_path.as_posix(), 0)


# Morphological Thinning
def thinning(image):
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, opened)
        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return skel

# Object Filtering


def filter_objects(image, size_threshold):
    output = np.zeros_like(image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
            output[labels == i] = 255
    return output

# Backbone Representation


def backbone_points(image, spacing=10):
    coords = np.column_stack(np.where(image > 0))
    sampled_coords = coords[::spacing]
    return sampled_coords


def get_backbone_points(image):
    # Step 1: Morphological Thinning
    thinned_image = thinning(image)
    # Step 2: Object Filtering
    size_threshold = 100  # Change this based on your application
    filtered_image = filter_objects(thinned_image, size_threshold)
    # Step 3: Backbone Representation
    spacing = 10  # Spacing in pixels
    points = backbone_points(filtered_image, spacing)

    # Flip row and column for each point to match Cartesian system
    points = points[:, [1, 0]]

    return points, {"thinned_image": thinned_image, "filtered_image": filtered_image}


if __name__ == "__main__":
    image = read_segmented_image(DATA_DIR / "14_top.png")
    points, info = get_backbone_points(image)

    image_w_points = plot_over_image(image, points, alternate_color=True)
    plt.imshow(image_w_points)
    plt.show()

    print("Sampled backbone points:", points)
