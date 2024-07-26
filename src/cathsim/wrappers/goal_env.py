import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GoalEnvWrapper(gym.ObservationWrapper):
    """Wraps a Gym environment into a GoalEnv"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._env = env

        self.observation_space = gym.spaces.Dict(
            **self._env.observation_space.spaces,
            desired_goal=spaces.Box(
                low=-np.inf, high=np.inf, shape=self._env.target.shape, dtype=np.float32
            ),
            achieved_goal=spaces.Box(
                low=-np.inf, high=np.inf, shape=self._env.target.shape, dtype=np.float32
            ),
        )

    def observation(self, observation):
        """Augment the observation with the goal."""
        obs = observation.copy()
        obs["desired_goal"] = self.goal
        obs["achieved_goal"] = self._env.head_pos
        return obs

    @property
    def goal(self):
        """The goal property."""
        return self._env.target

    def set_goal(self, goal):
        """Set the goal property."""
        self._env.set_target(goal)
