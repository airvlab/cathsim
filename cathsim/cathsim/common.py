import yaml
import numpy as np
from pathlib import Path

config_path = Path(__file__).parent / 'env_config.yaml'
with open(config_path.as_posix()) as f:
    env_config = yaml.safe_load(f)


def normalize_rgba(rgba: list):
    new_rgba = [c / 255. for c in rgba]
    new_rgba[-1] = rgba[-1]
    return new_rgba


def point2pixel(point, camera_matrix: np.ndarray = None):
    """Transforms from world coordinates to pixel coordinates."""
    assert len(point) == 3, 'Point must be a 3D vector.'
    assert camera_matrix is not None, 'Camera matrix must be provided.'
    x, y, z = point
    xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

    return np.array([round(xs / s), round(ys / s)]).asttype(np.int8)


def filter_mask(segment_image: np.ndarray):
    geom_ids = segment_image[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    segment_image = 255 * geom_ids
    return segment_image
