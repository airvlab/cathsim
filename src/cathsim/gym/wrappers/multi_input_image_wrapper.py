import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


class MultiInputImageWrapper(gym.ObservationWrapper):
    """An image wrapper for MultiInput environments that have images as one of the observation components.

    This wrapper allows for:
    - Conversion of images to grayscale.
    - Option to keep channel dimension even for grayscale images.
    - Option to place channel either first or last in image shape.

    Args:
        env (gym.Env): The environment to wrap.
        grayscale (bool): Whether to convert the image to grayscale. Defaults to False.
        keep_dim (bool): Whether to keep the channel dimension after conversion to grayscale. Defaults to True.
        channel_first (bool): Whether to place channel first in shape. Defaults to False.
        image_key (str): The key of the image in the observation dictionary. Defaults to "pixels".
    """

    def __init__(
        self,
        env: gym.Env,
        grayscale: bool = False,
        keep_dim: bool = True,
        channel_first: bool = False,
        image_key: str = "pixels",
    ):
        super(MultiInputImageWrapper, self).__init__(env)
        self.grayscale = grayscale
        self.image_key = image_key
        self.keep_dim = keep_dim
        self.channel_first = channel_first

        image_space = self.observation_space.spaces[self.image_key]
        h, w, c = image_space.shape

        assert c == 3, "Image should be in RGB format with 3 channels."

        if self.grayscale:
            if self.keep_dim:
                new_shape = (1, h, w) if self.channel_first else (h, w, 1)
            else:
                new_shape = (h, w)
            image_space = spaces.Box(
                low=0, high=255, shape=new_shape, dtype=image_space.dtype
            )

        self.observation_space.spaces[self.image_key] = image_space

    def observation(self, observation):
        image = observation[self.image_key]
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                image = np.expand_dims(image, axis=0 if self.channel_first else -1)
        observation[self.image_key] = image
        return observation


if __name__ == "__main__":
    from cathsim.gym.envs import CathSim, make_gym_env
    from cathsim.rl.utils import Config

    config = Config("pixels")
    env = make_gym_env(config, n_envs=1, monitor_wrapper=True)

    for k, v in env.observation_space.spaces.items():
        print(k, v.shape)
