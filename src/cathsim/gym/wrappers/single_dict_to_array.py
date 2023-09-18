import gymnasium as gym


class SingleDict2Array(gym.ObservationWrapper):
    """Wrapper for dm_control environments to be used with OpenAI gym."""

    def __init__(self, env):
        super(SingleDict2Array, self).__init__(env)
        self.observation_space = next(iter(self.observation_space.values()))

    def observation(self, observation):
        """Unpack the observation dictionary."""
        obs = next(iter(observation.values()))
        return obs
