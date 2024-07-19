import pytest
from dm_control import composer
from dm_control.mujoco import engine
import numpy as np
from cathsim.dm import Phantom, Guidewire, Tip, Navigate
from unittest.mock import Mock


@pytest.fixture
def task():
    phantom_name = 'phantom4'
    phantom = Phantom(phantom_name + '.xml')
    tip = Tip()
    guidewire = Guidewire()

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target='bca',
        sample_target=True,
        image_size=480,
        visualize_sites=True,
    )
    return task


def test_initialize_episode(task):
    mock_physics = Mock()
    mock_random_state = np.random.RandomState(42)
    task.initialize_episode(mock_physics, mock_random_state)
    assert task.success is False


def test_initialize_episode_mjcf(task):
    mock_random_state = np.random.RandomState(42)
    task.initialize_episode_mjcf(mock_random_state)
    assert task._mjcf_variator is not None


def test_should_terminate_episode(task):
    task.success = True
    mock_physics = Mock()
    assert task.should_terminate_episode(mock_physics) is True

    task.success = False
    assert task.should_terminate_episode(mock_physics) is False


@pytest.mark.parametrize("image_size", [80, 480])
@pytest.mark.parametrize("camera_name", ["top_camera", "side"])
def test_camera_matrix(task, image_size, camera_name):
    name2id = {"top_camera": 0, "side": 2}
    task.get_camera_matrix(image_size, camera_name)

    env = composer.Environment(task)
    physics = env.physics
    camera = engine.Camera(physics, height=image_size, width=image_size,
                           camera_id=name2id[camera_name])
    __import__('pprint').pprint(task.camera_matrix)
    __import__('pprint').pprint(camera.matrix)
    assert np.allclose(task.camera_matrix, camera.matrix), \
        "Camera matrix is not correct"
