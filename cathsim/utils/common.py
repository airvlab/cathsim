from toolz.dicttoolz import itemmap
import numpy as np
import yaml
from pathlib import Path


def flatten_dict(d: dict, parent_key: str = None) -> dict:
    acc = {}
    for k, v in d.items():
        if parent_key:
            k = parent_key + "-" + k
        if isinstance(v, dict):
            acc = acc | flatten_dict(v, k)
        else:
            acc[k] = v
    return acc


def expand_dict(xd: dict, yd: dict) -> dict:
    zd = xd.copy()
    for k, v in yd.items():
        if k not in xd:
            zd[k] = [v]
        elif isinstance(v, dict) and isinstance(xd[k], dict):
            zd[k] = expand_dict(xd[k], v)
        else:
            zd[k] = xd[k] + [v]
    return zd


def map_val(g: callable, d: dict):
    def f(item):
        k, v = item
        if isinstance(v, dict):
            return (k, itemmap(f, v))
        else:
            return (k, g(v))

    return itemmap(f, d)


def normalize_rgba(rgba: list):
    new_rgba = [c / 255.0 for c in rgba]
    new_rgba[-1] = rgba[-1]
    return new_rgba


def point2pixel(point, camera_kwargs: dict = dict(image_size=80)):
    """Transforms from world coordinates to pixel coordinates."""
    camera_matrix = create_camera_matrix(**camera_kwargs)
    x, y, z = point
    xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

    return np.array([round(xs / s), round(ys / s)]).astype(np.int32)


def filter_mask(segment_image: np.ndarray):
    geom_ids = segment_image[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    segment_image = 255 * geom_ids
    return segment_image


def create_camera_matrix(
    image_size, pos=np.array([-0.03, 0.125, 0.15]), euler=np.array([0, 0, 0]), fov=45
) -> np.ndarray:
    def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
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


def get_env_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "env_config.yaml"
    with open(config_path, "r") as f:
        env_config = yaml.safe_load(f)
    return env_config
