import numpy as np
from cathsim.gym.envs import CathSim
import gymnasium as gym
import pytest

from cathsim.gym.wrappers import (
    MultiInputImageWrapper,
)


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
