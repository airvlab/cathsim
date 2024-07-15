import numpy as np
import cv2
import gymnasium as gym


class MultiInputImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        grayscale: bool = False,
        keep_dim: bool = True,
        channel_first: bool = False,
        image_key: str = "pixels",
    ):
        super().__init__(env)
        self.grayscale = grayscale
        self.keep_dim = keep_dim
        self.channel_first = channel_first
        self.image_key = image_key

        image_space = self.observation_space.spaces[self.image_key]
        h, w, c = image_space.shape
        assert c == 3, "Image should be in RGB format with 3 channels."

        # Handle grayscale option
        if self.grayscale:
            new_shape = (
                ((1, h, w) if self.channel_first else (h, w, 1))
                if self.keep_dim
                else (h, w)
            )
            image_space = gym.spaces.Box(
                0, 255, shape=new_shape, dtype=image_space.dtype
            )

        # Handle channel_first option
        elif self.channel_first:
            new_shape = (c, h, w)
            image_space = gym.spaces.Box(
                0, 255, shape=new_shape, dtype=image_space.dtype
            )

        self.observation_space.spaces[self.image_key] = image_space

    def observation(self, observation):
        image = observation[self.image_key]

        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                axis = 0 if self.channel_first else -1
                image = np.expand_dims(image, axis=axis)

        elif self.channel_first:  # Only reshape if grayscale conversion is not done
            image = np.transpose(image, (2, 0, 1))

        observation[self.image_key] = image
        return observation


if __name__ == "__main__":
    from cathsim.gym.envs import CathSim, make_gym_env
    from cathsim.rl.utils import Config

    config = Config("pixels")
    env = make_gym_env(config, n_envs=1, monitor_wrapper=True)

    for k, v in env.observation_space.spaces.items():
        print(k, v.shape)
