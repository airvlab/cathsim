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

        self.f = f
        self.key = key
        self.new_shape = new_shape

    def observation(self, observation: dict):
        if self.key in observation:
            transformed_observation = self.f(observation[self.key])

            if self.new_shape == "auto":
                new_shape = transformed_observation.shape
            else:
                new_shape = self.new_shape

            # Update observation space to reflect new shape
            original_space = self.observation_space.spaces[self.key]
            self.observation_space.spaces[self.key] = gym.spaces.Box(
                low=original_space.low.min(),
                high=original_space.high.max(),
                shape=new_shape,
                dtype=original_space.dtype,
            )

            observation[self.key] = transformed_observation

        return observation
