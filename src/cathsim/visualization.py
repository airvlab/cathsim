import matplotlib.pyplot as plt
import numpy as np
import warnings


def create_camera_matrix(
    image_size: int,
    pos=np.array([-0.03, 0.125, 0.15]),
    euler=np.array([0, 0, 0]),
    fov=45,
) -> np.ndarray:
    """
    Generate a camera matrix given the parameters

    Args:
      image_size: int:
      pos:  (Default value = np.array([-0.03, 0.125, 0.15]):
      euler:  (Default value = np.array([0, 0, 0]):
      fov:  (Default value = 45)

    Returns:
        np.ndarray: The camera matrix

    """

    def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        Make a rotation matrix from euler angles

        Args:
            euler_angles: np.ndarray: The euler angles

        Returns:
            np.ndarray: The rotation matrix

        """
        rx, ry, rz = np.deg2rad(euler_angles)

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        R = Rz @ Ry @ Rx
        return R

    # Intrinsic Parameters
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * image_size / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    image = np.eye(3)
    image[0, 2] = (image_size - 1) / 2.0
    image[1, 2] = (image_size - 1) / 2.0

    # Extrinsic Parameters
    rotation_matrix = euler_to_rotation_matrix(euler)
    R = np.eye(4)
    R[0:3, 0:3] = rotation_matrix
    T = np.eye(4)
    T[0:3, 3] = -pos

    # Camera Matrix
    camera_matrix = image @ focal @ R @ T
    return camera_matrix


def point2pixel(
    point: np.ndarray,
    camera_matrix: np.ndarray = None,
    camera_kwargs: dict = dict(image_size=80),
) -> np.ndarray:
    """Transforms from world coordinates to pixel coordinates.

    Args:
      point: np.ndarray: the point to be transformed.
      camera_matrix: np.ndarray: the camera matrix. If None, it is generated from the camera_kwargs.
      camera_kwargs: dict:  (Default value = dict(image_size=80))

    Returns:
        np.ndarray : the pixel coordinates of the point.
    """
    if camera_matrix is None:
        camera_matrix = create_camera_matrix(**camera_kwargs)
    x, y, z = point
    xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

    return np.array([round(xs / s), round(ys / s)]).astype(np.int32)


def plot_3D_to_2D(
    ax: plt.Axes,
    data: np.ndarray,
    base_image: np.ndarray = None,
    add_line: bool = True,
    image_size: int = 80,
    line_kwargs: dict = dict(color="black"),
    scatter_kwargs: dict = dict(color="blue"),
) -> plt.Axes:
    """Plot 3D data to 2D image

    Args:
        ax (plt.Axes): The axes to plot on
        data (np.ndarray): The data to plot
        base_image (np.ndarray): An image to plot behind the data
        add_line (bool): If True, add a line between the points
        image_size (int): Size of the image. If base_image is provided, this is overwritten
        line_kwargs (dict): Keyword arguments for the line. See plt.plot
        scatter_kwargs (dict): Keyword arguments for the scatter. See plt.scatter
    """
    if base_image is not None:
        base_image = np.flipud(base_image)
        plt.imshow(base_image)
        image_size = base_image.shape[0]
        warnings.warn("Image size overwritten by base_image")

    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    if len(data[0]) == 3:
        data = np.apply_along_axis(
            point2pixel, 1, data, camera_kwargs=dict(image_size=image_size)
        )
        data = [point for point in data if np.all((0 <= point) & (point <= image_size))]
        data = np.array(data)  # Convert back to numpy array
        data[:, 1] = image_size - data[:, 1]
    ax.scatter(data[:, 0], data[:, 1], **scatter_kwargs)
    if add_line:
        ax.plot(data[:, 0], data[:, 1], **line_kwargs)


def plot_w_mesh(mesh, points: np.ndarray, **kwargs):
    """
    Plot a mesh with points

    Args:
        mesh: The mesh to plot
        points: The points to plot
        **kwargs: The keyword arguments to pass to the scatter function

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    ax.scatter(points[:, 0], points[:, 0], points[:, 0], **kwargs)
    plt.show()
