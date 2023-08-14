import numpy as np
import cv2

from gym.envs.registration import EnvSpec
from gym import spaces
import gym

from dm_control import composer
from dm_env import specs


def convert_dm_control_to_gym_space(dm_control_space):
    """Convert dm_control space to gym space."""
    if isinstance(dm_control_space, specs.BoundedArray):
        low, high = (
            (0, 255)
            if len(dm_control_space.shape) > 1
            else (dm_control_space.minimum, dm_control_space.maximum)
        )
        return spaces.Box(
            low=low,
            high=high,
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype
            if len(dm_control_space.shape) > 1
            else np.float32,
        )

    if isinstance(dm_control_space, specs.Array) and not isinstance(
        dm_control_space, specs.BoundedArray
    ):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=np.float32,
        )

    if isinstance(dm_control_space, dict):
        return spaces.Dict(
            {
                key: convert_dm_control_to_gym_space(value)
                for key, value in dm_control_space.items()
            }
        )


class DMEnvToGymWrapper(gym.Env):
    """Wrapper for dm_control environments to be used with OpenAI gym."""

    spec = EnvSpec("CathSim-v0", max_episode_steps=300)

    def __init__(self, env: composer.Environment, env_kwargs: dict = {}) -> gym.Env:
        self._env = env
        self.metadata = {
            "render.modes": ["rgb_array"],
            "video.frames_per_second": round(1.0 / self._env.control_timestep()),
        }

        self.env_kwargs = env_kwargs
        self.image_size = self._env.task.image_size

        self.action_space = convert_dm_control_to_gym_space(
            self._env.action_spec(),
        )
        self.observation_space = convert_dm_control_to_gym_space(
            self._env.observation_spec(),
        )

        self.viewer = None
        self.use_contact_forces = False
        self.use_force = False
        self.use_geom_pos = False
        # self.goal = np.array([0.0, 0.0, 0.0])

    def seed(self, seed):
        return self._env.random_state.seed

    def step(self, action):
        timestep = self._env.step(action)
        observation = self._get_obs(timestep)
        reward = timestep.reward
        done = timestep.last()
        info = dict(
            head_pos=self.head_pos.copy(),
            target_pos=self.target.copy(),
        )
        if self.use_contact_forces:
            info["contact_forces"] = self.contact_forces.copy()
        if self.use_geom_pos:
            info["geom_pos"] = self.guidewire_geom_pos.copy()
        return observation, reward, done, info

    def reset(self):
        timestep = self._env.reset()
        obs = self._get_obs(timestep)
        return obs

    def render(self, mode="rgb_array", image_size=None):
        image_size = image_size if image_size else self.image_size
        img = self._env.physics.render(height=image_size, width=image_size, camera_id=0)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def _get_obs(self, timestep):
        obs = timestep.observation
        for key, value in obs.items():
            if value.dtype == np.float64:
                obs[key] = value.astype(np.float32)
        return obs

    @property
    def head_pos(self):
        return self._env._task.get_head_pos(self.physics)

    @property
    def force(self):
        return self._env._task.get_force(self.physics)

    @property
    def contact_forces(self):
        return self._env._task.get_contact_forces(self.physics, self.image_size)

    @property
    def physics(self):
        return self._env._physics.copy()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._env._task.compute_reward(achieved_goal, desired_goal)

    @property
    def target(self):
        """The goal property."""
        return self._env._task.target_pos

    def set_target(self, goal):
        self._env._task.set_target(goal)

    @property
    def guidewire_geom_pos(self):
        """The guidewire_geom_pos property."""
        return self._env._task.get_guidewire_geom_pos(self.physics)


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
        obs = observation.copy()
        obs["desired_goal"] = self.goal
        obs["achieved_goal"] = self._env.head_pos
        return obs

    @property
    def goal(self):
        """The goal property."""
        return self._env.target

    def set_goal(self, goal):
        self._env.set_target(goal)


class Dict2Array(gym.ObservationWrapper):
    def __init__(self, env):
        super(Dict2Array, self).__init__(env)
        self.observation_space = next(iter(self.observation_space.values()))

    def observation(self, observation):
        obs = next(iter(observation.values()))
        return obs


class MultiInputImageWrapper(gym.ObservationWrapper):
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

        assert (
            len(image_space.shape) == 3 and image_space.shape[-1] == 3
        ), "Image should be in RGB format"

        if self.grayscale:
            if self.keep_dim:
                if self.channel_first:
                    shape = (1, image_space.shape[0], image_space.shape[1])
                else:
                    shape = (image_space.shape[0], image_space.shape[1], 1)
                image_space = spaces.Box(
                    low=0, high=255, shape=shape, dtype=image_space.dtype
                )
            else:
                image_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(image_space.shape[0], image_space.shape[1]),
                    dtype=image_space.dtype,
                )

        self.observation_space[self.image_key] = image_space

    def observation(self, observation):
        image = observation[self.image_key]
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                image = np.expand_dims(image, axis=0 if self.channel_first else -1)
        observation[self.image_key] = image
        return observation
