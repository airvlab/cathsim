from typing import Any, Callable, Dict
import gymnasium as gym
import numpy as np
import cv2


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


if __name__ == "__main__":
    from cathsim.gym.envs import CathSim

    def transform_image(observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(observation, axis=-1)

    env = gym.make("cathsim/CathSim-v0", use_pixels=True)
    print("Original observation space:")
    for key, value in env.observation_space.spaces.items():
        print("\t", key, value.shape)
    env = TransformObservationForKey(env, transform_image, "pixels")
    print("Transformed observation space:")
    for key, value in env.observation_space.spaces.items():
        print("\t", key, value.shape)
