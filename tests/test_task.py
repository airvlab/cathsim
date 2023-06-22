import pytest
from dm_control import composer
import numpy as np
from cathsim.cathsim.env import Phantom, Guidewire, Tip, Navigate
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
    assert task.success == False


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
