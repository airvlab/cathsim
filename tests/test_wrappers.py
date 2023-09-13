import numpy as np
import cv2
from cathsim.gym.envs import CathSim
import gymnasium as gym
import pytest

from cathsim.gym.wrappers import MultiInputImageWrapper, TransformDictObservation


def check_obs_space_discrepancy(obs, env):
    if isinstance(obs, gym.spaces.Box):
        assert obs.shape == env.shape, f"shape: {obs.shape}"
        assert obs.dtype == env.dtype, f"dtype: {obs.dtype}"
    elif isinstance(obs, gym.spaces.Dict):
        for key in obs.spaces.keys():
            check_obs_space_discrepancy(obs.spaces[key], env[key])


class TestMultiInputImageWrapper:
    _image_size = 80

    @pytest.fixture
    def dummy_env(self):
        return gym.make(
            "cathsim/CathSim-v0", use_pixels=True, image_size=self._image_size
        )

    def test_default_initialization(self, dummy_env):
        env = MultiInputImageWrapper(dummy_env)
        obs, _ = env.reset()
        assert "pixels" in obs.keys()
        assert obs["pixels"].shape == (80, 80, 3)

    def test_grayscale(self, dummy_env):
        env = MultiInputImageWrapper(dummy_env, grayscale=True)
        obs, _ = env.reset()
        assert obs["pixels"].shape == (80, 80, 1)

    def test_grayscale_without_dim(self, dummy_env):
        env = MultiInputImageWrapper(dummy_env, grayscale=True, keep_dim=False)
        obs, _ = env.reset()
        assert obs["pixels"].shape == (80, 80)

    def test_channel_first(self, dummy_env):
        env = MultiInputImageWrapper(dummy_env, channel_first=True)
        obs, _ = env.reset()
        assert obs["pixels"].shape == (3, 80, 80)

    def test_grayscale_channel_first(self, dummy_env):
        env = MultiInputImageWrapper(dummy_env, grayscale=True, channel_first=True)
        obs, _ = env.reset()
        assert obs["pixels"].shape == (1, 80, 80), f"shape: {obs['pixels'].shape}"


class TestTransformDictObservation:
    @pytest.fixture
    def dummy_env(self):
        return gym.make("cathsim/CathSim-v0", use_pixels=True)

    def test_add_dimension(self, dummy_env):
        def add_dimension(obs):
            return np.expand_dims(obs, axis=-1)

        env = TransformDictObservation(dummy_env, add_dimension, "pixels")
        obs, _ = env.reset()
        check_obs_space_discrepancy(obs, env)
        assert obs["pixels"].shape == (80, 80, 3, 1)

    def test_grayscale(self, dummy_env):
        def grayscale(obs):
            return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        env = TransformDictObservation(dummy_env, grayscale, "pixels")
        obs, _ = env.reset()
        assert obs["pixels"].shape == (80, 80)

    def test_manual_new_shape(self, dummy_env):
        def reshape(obs):
            return obs.reshape((10, 10))

        env = TransformDictObservation(dummy_env, reshape, "pixels", new_shape=(10, 10))

        assert env.observation_space["pixels"].shape == (10, 10)
