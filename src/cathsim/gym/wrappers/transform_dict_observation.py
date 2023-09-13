from typing import Any, Callable, Union
import gymnasium as gym


class TransformDictObservation(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        f: Callable[[Any], Any],
        key: str,
        new_shape: Union[tuple, str] = "auto",
    ):
        super().__init__(env)

        assert callable(f), "`f` needs to be a callable function."
        assert isinstance(key, str), "`key` needs to be a string."
        assert (
            key in self.observation_space.spaces
        ), f"Key {key} not in observation space."

        self.f = f
        self.key = key
        self.new_shape = new_shape

        if self.key in self.observation_space.spaces:
            original_space = self.observation_space.spaces[self.key]

            if self.new_shape == "auto":
                sample_transformed_observation = self.f(original_space.sample())
                new_shape = sample_transformed_observation.shape
            else:
                new_shape = self.new_shape

            self.observation_space.spaces[self.key] = gym.spaces.Box(
                low=original_space.low.min(),
                high=original_space.high.max(),
                shape=new_shape,
                dtype=original_space.dtype,
            )

    def observation(self, observation: dict):
        if self.key in observation:
            observation[self.key] = self.f(observation[self.key])
        return observation


if __name__ == "__main__":
    import cv2
    from cathsim.gym.envs import CathSim

    def transform_image(observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

    env = gym.make("cathsim/CathSim-v0", use_pixels=True)
    env.unwrapped.print_spaces()
    obs, _ = env.reset()
    env = TransformDictObservation(env, transform_image, "pixels")
    obs, _ = env.reset()
    print(obs["pixels"].shape)
    env.unwrapped.print_spaces()
