"""Wrappers for dm_control environments to be used with OpenAI gym."""
import numpy as np
import cv2

from gym.envs.registration import EnvSpec
from gym import spaces
import gym

import dm_control
from dm_control import composer
from dm_env import specs
import dm_env

from typing import Optional, Dict


def convert_dm_control_to_gym_space(dm_control_space: dm_env.specs) -> gym.spaces:
    """Convert DM control space to Gym space.

    Args:
        dm_control_space (dm_env.specs): The DM control space.

    Returns:
        gym.spaces: The Gym space.
    """
    # If dm_control_space is an instance of BoundedArray.
    if isinstance(dm_control_space, specs.BoundedArray):
        low, high = (
            (0, 255)
            if len(dm_control_space.shape) > 1
            else (dm_control_space.minimum, dm_control_space.maximum)
        )
        return spaces.Box(
            low=np.float32(low),
            high=np.float32(high),
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype
            if len(dm_control_space.shape) > 1
            else np.float32,
        )

    # If dm_control_space is an instance of Array (but not BoundedArray).
    elif isinstance(dm_control_space, specs.Array):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=np.float32,
        )

    # If dm_control_space is a dictionary.
    elif isinstance(dm_control_space, dict):
        return spaces.Dict(
            {
                key: convert_dm_control_to_gym_space(value)
                for key, value in dm_control_space.items()
            }
        )

    # Consider raising an error if none of the types match for clearer error handling.
    else:
        raise ValueError(f"Unsupported DM control space type: {type(dm_control_space)}")


class DMEnvToGymWrapper(gym.Env):
    """Wrapper for dm_control environments to be used with OpenAI gym."""

    spec = EnvSpec("CathSim-v0", max_episode_steps=300)

    def __init__(
        self,
        env: composer.Environment,
        use_contact_forces: bool = False,
        use_force: bool = True,
        use_geom_pos: bool = False,
    ):
        """Initialize the wrapper.

        Args:
            env (composer.Environment): DM control environment.
            use_contact_forces (bool): Use contact forces or not. Defaults to False.
            use_force (bool): Use force or not. Defaults to True.
            use_geom_pos (bool): Use geometry position or not. Defaults to False.
        """
        self._env = env
        self.metadata = {
            "render.modes": ["rgb_array"],
            "video.frames_per_second": round(1.0 / self._env.control_timestep()),
        }

        self.image_size = self._env.task.image_size

        self.action_space = convert_dm_control_to_gym_space(
            self._env.action_spec(),
        )
        self.observation_space = convert_dm_control_to_gym_space(
            self._env.observation_spec(),
        )

        self.viewer = None
        self.use_contact_forces = use_contact_forces
        self.use_force = use_force
        self.use_geom_pos = use_geom_pos

    def seed(self, seed: int):
        """Environment seed. This is used to seed the environment to get reproducible results."""
        return self._env.random_state.seed

    def step(self, action: np.ndarray) -> tuple:
        """Take a step in the environment.

        Args:
            action (np.ndarray): The action to take in the environment.
        """

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

        if self.use_force:
            info["forces"] = self.force.copy()

        if self.use_geom_pos:
            info["geom_pos"] = self.guidewire_geom_pos.copy()
        return observation, reward, done, info

    def reset(self) -> gym.spaces:
        """Reset the environment.

        Returns:
            gym.spaces: The initial observation of the environment.
        """

        timestep = self._env.reset()
        obs = self._get_obs(timestep)
        return obs

    def render(self, mode: str = "rgb_array", image_size: int = None) -> np.ndarray:
        """Render the environment.

        Args:
            mode: can be one of ['rgb_array', 'human'] Note: 'human' mode is not supported.
            image_size: The size of the image to render. Defaults to self.image_size.

        Returns:
            np.ndarray: The rendered image.
        """

        image_size = image_size if image_size else self.image_size
        img = self._env.physics.render(height=image_size, width=image_size, camera_id=0)
        return img

    def close(self):
        """ """
        # Close the viewer.
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def _get_obs(self, timestep):
        """Get the observation from the dm_env timestep.

        Args:
            timestep: The timestep from the dm_env.
        """

        obs = timestep.observation
        for key, value in obs.items():
            if value.dtype == np.float64:
                obs[key] = value.astype(np.float32)
        return obs

    @property
    def head_pos(self) -> np.ndarray:
        """Get the position of the guidewire tip."""
        return self._env._task.get_head_pos(self.physics)

    @property
    def force(self) -> np.ndarray:
        """The magnitude of the force applied to the aorta."""
        return self._env._task.get_total_force(self.physics)

    @property
    def contact_forces(self) -> np.ndarray:
        """Get the contact forces for each contact point."""
        return self._env._task.get_contact_forces(self.physics, self.image_size)

    @property
    def physics(self) -> dm_control.mujoco.engine.Physics:
        """Returns Physics object that is associated with the dm_env."""
        return self._env._physics.copy()

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = {}
    ) -> np.ndarray:
        """Compute the reward based on the achieved goal and the desired goal.

        Args:
            achieved_goal: The achieved goal.
            desired_goal: The desired goal.
            info: Additional information. Currently not used.

        Returns:
            np.ndarray: The reward.
        """
        return self._env._task.compute_reward(achieved_goal, desired_goal)

    @property
    def target(self) -> np.ndarray:
        """The target position."""
        return self._env._task.target_pos

    def set_target(self, goal: np.ndarray):
        """Set the target position."""
        self._env._task.set_target(goal)

    @property
    def guidewire_geom_pos(self) -> np.ndarray:
        """The position of the guidewire body geometries. This property is used to determine the shape of the guidewire."""
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


class Dict2Array(gym.ObservationWrapper):
    """Wrapper for dm_control environments to be used with OpenAI gym."""

    def __init__(self, env):
        super(Dict2Array, self).__init__(env)
        self.observation_space = next(iter(self.observation_space.values()))

    def observation(self, observation):
        """Unpack the observation dictionary."""
        obs = next(iter(observation.values()))
        return obs


class MultiInputImageWrapper(gym.ObservationWrapper):
    """An image wrapper for MultiInputImageWrapper.

    Args:
        env: The environment to wrap.
        grayscale: Whether to convert the image to grayscale.
        keep_dim: Whether to keep the dimension of the image.
        channel_first: Whether to keep the channel first or last.
        image_key: The key of the image in the observation dictionary.
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
