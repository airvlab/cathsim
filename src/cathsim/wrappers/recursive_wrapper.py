from typing import Any, Callable, Dict

import cv2
import gymnasium as gym
import numpy as np


class TransformObservationForKey(
    gym.ObservationWrapper, gym.utils.RecordConstructorArgs
):
    """Transform the observation of a specific key via an arbitrary function :attr:`f`."""

    def __init__(self, env: gym.Env, f: Callable[[Any], Any], key: str):
        """
        Initialize the :class:`TransformObservationForKey` wrapper with an environment, a transform function :attr:`f`,
        and a specific key.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
            key: The specific key in the observation dictionary to transform
        """
        gym.utils.RecordConstructorArgs.__init__(self, f=f, key=key)
        gym.ObservationWrapper.__init__(self, env)

        assert callable(f)
        self.f = f
        self.key = key

        # Update the observation space for the specific key if it exists
        if self.key in self.observation_space.spaces:
            transformed_space = self.f(self.observation_space.spaces[self.key].sample())
            self.observation_space.spaces[self.key] = gym.spaces.Box(
                low=0, high=255, shape=transformed_space.shape, dtype=np.uint8
            )

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms the observations of a specific key with callable :attr:`f`.

        Args:
            observation: The observation dictionary to transform

        Returns:
            The transformed observation dictionary
        """
        if self.key in observation:
            observation[self.key] = self.f(observation[self.key])
        return observation
