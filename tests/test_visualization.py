import pytest
from pprint import pformat
import numpy as np

from cathsim.dm import visualization, make_dm_env
from dm_control import mujoco
from scipy.spatial.transform import Rotation as R

_image_size = 80


@pytest.fixture
def dm_env():
    return make_dm_env("phantom3", use_pixels=True)


def matrix4x4_to_quat(matrix_4x4):
    rotation_matrix_3x3 = matrix_4x4[0:3, 0:3]
    r = R.from_matrix(rotation_matrix_3x3)
    quat = r.as_quat()

    return quat


def translation_matrix_to_vec(translation_matrix):
    return translation_matrix[0:3, 3]


def get_camera(dm_env, camera_id):
    cameras = dm_env._task._arena.mjcf_model.find_all("camera")
    return cameras[camera_id]


@pytest.mark.parametrize("image_size, camera_id", [(480, 2)])
def test_create_camera_matrix(dm_env, image_size, camera_id):
    physics = dm_env.physics

    camera_expected = mujoco.Camera(
        physics,
        height=image_size,
        width=image_size,
        camera_id=camera_id
    )
    camera_actual = get_camera(dm_env, camera_id)

    assert (camera_expected is not None)
    assert (camera_actual is not None)

    matrices_expected = camera_expected.matrices()
    image_expected = matrices_expected.image
    focal_expected = matrices_expected.focal
    rotation_expected = matrices_expected.rotation
    translation_expected = matrices_expected.translation
    quat_expected = matrix4x4_to_quat(rotation_expected)

    quat_actual = camera_actual.quat
    pos_actual = camera_actual.pos
    print("quat_actual: ", quat_actual)
    print("quat_expected: ", quat_actual)

    # quat_actual = [quat_actual[1], quat_actual[2], quat_actual[3], quat_actual[0]]
    rotation_actual = R.from_quat(quat_actual).as_matrix()

    quat_expected = np.array(quat_expected)
    quat_actual = np.array(quat_actual)

    translation_actual = np.eye(4)
    translation_actual[0:3, 3] = -pos_actual

    assert np.allclose(quat_expected, quat_actual) or np.allclose(quat_expected, -quat_actual)
    assert np.allclose(rotation_expected[:3, :3], rotation_actual) or np.allclose(rotation_expected[:3, :3], -rotation_actual)
    assert np.allclose(translation_expected, translation_actual)

    matrix_expected = visualization.create_camera_matrix(
        image_size=image_size,
        pos=pos_actual,
        R=rotation_actual,
    )

    assert np.allclose(matrix_expected, camera_expected.matrix)
