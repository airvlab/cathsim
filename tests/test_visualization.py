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
    matrix_4x4 = matrix_4x4.T
    rotation_matrix_3x3 = matrix_4x4[0:3, 0:3]
    r = R.from_matrix(rotation_matrix_3x3)
    quat = r.as_quat()
    quat = [quat[3], quat[0], quat[1], quat[2]]

    return quat


def translation_matrix_to_vec(translation_matrix):
    return translation_matrix[0:3, 3]


def get_camera(dm_env, camera_id):
    cameras = dm_env._task._arena.mjcf_model.find_all("camera")
    return cameras[camera_id]


@pytest.mark.parametrize("image_size, camera_id", [(480, 0), (480, 2)])
def test_create_camera_matrix(dm_env, image_size, camera_id):
    physics = dm_env.physics

    camera_expected = mujoco.Camera(physics, height=image_size, width=image_size, camera_id=camera_id)
    camera_actual = get_camera(dm_env, camera_id)

    assert (camera_expected is not None)
    assert (camera_actual is not None)

    matrices_expected = camera_expected.matrices()
    rotation_expected = matrices_expected.rotation
    rotation_expected = rotation_expected[0:3, 0:3]
    translation_expected = matrices_expected.translation
    quat_expected = matrix4x4_to_quat(rotation_expected)
    pos_expected = -translation_matrix_to_vec(translation_expected)
    matrix_expected = camera_expected.matrix

    quat_actual = camera_actual.quat
    pos_actual = camera_actual.pos
    rotation_actual = visualization.quat_to_mat(quat_actual)
    translation_actual = np.eye(4)
    translation_actual[0:3, 3] = -pos_actual

    assert np.allclose(pos_expected, pos_actual)
    assert np.allclose(quat_expected, quat_actual) or np.allclose(quat_expected, -quat_actual)

    print("rotation_actual: \n", rotation_actual)
    print("rotation_expected: \n", rotation_expected)
    print("quat_actual: ", quat_actual)
    print("quat_expected: ", quat_expected)

    assert np.allclose(rotation_expected, rotation_actual) or np.allclose(rotation_expected, -rotation_actual)
    assert np.allclose(translation_expected, translation_actual)

    matrix_actual = visualization.create_camera_matrix(
        image_size=image_size,
        pos=pos_actual,
        quat=quat_actual,
        debug=True,
    )
    print("matrix_expected: \n", matrix_actual)
    print("matrix_actual: \n", matrix_expected)

    assert np.allclose(matrix_expected, matrix_actual)
