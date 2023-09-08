"""Wrappers for dm_control environments to be used with OpenAI gym."""
import numpy as np

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.envs.registration import EnvSpec

from dm_env import specs
from cathsim.dm import make_dm_env


def convert_spec_to_gym_space(dm_control_space: specs) -> gym.spaces:
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

    elif isinstance(dm_control_space, specs.Array):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=np.float32,
        )

    elif isinstance(dm_control_space, dict):
        return spaces.Dict(
            {
                key: convert_spec_to_gym_space(value)
                for key, value in dm_control_space.items()
            }
        )

    else:
        raise ValueError(f"Unsupported DM control space type: {type(dm_control_space)}")


class CathSim(gym.Env):
    spec = EnvSpec("cathsim/CathSim-v0", max_episode_steps=300)

    def __init__(
        self,
        phantom: str = "phantom3",
        use_contact_forces: bool = False,
        use_force: bool = False,
        use_geom_pos: bool = False,
        **kwargs,
    ):
        self._env = make_dm_env(phantom=phantom, **kwargs)

        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": round(1.0 / self._env.control_timestep()),
        }

        self.image_size = self._env.task.image_size

        self.action_space = convert_spec_to_gym_space(
            self._env.action_spec(),
        )
        self.observation_space = convert_spec_to_gym_space(
            self._env.observation_spec(),
        )

        self.viewer = None
        self.use_contact_forces = use_contact_forces
        self.use_force = use_force
        self.use_geom_pos = use_geom_pos

    def _get_obs(self, timestep):
        obs = timestep.observation
        for key, value in obs.items():
            if value.dtype == np.float64:
                obs[key] = value.astype(np.float32)
        return obs

    def _get_info(self):
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
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        timestep = self._env.reset()
        obs = self._get_obs(timestep)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple:
        timestep = self._env.step(action)

        observation = self._get_obs(timestep)
        reward = timestep.reward
        terminated = timestep.last()
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render_frame(self, image_size=None, camera_id: int = 0):
        image_size = image_size or self.image_size
        img = self._env.physics.render(
            height=image_size, width=image_size, camera_id=camera_id
        )
        return img

    def render(self, mode: str = "rgb_array", image_size: int = None) -> np.ndarray:
        if mode == "rgb_array":
            return self.render_frame(image_size)
        else:
            raise NotImplementedError("Render mode '{}' is not supported.".format(mode))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self._env.close()

    def print_spaces(self):
        print("Observation space:")
        if isinstance(self.observation_space, spaces.Dict):
            for key, value in self.observation_space.spaces.items():
                print("\t", key, value.shape)
        print("Action space:")
        print("\t", self.action_space.shape)

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
    def physics(self):
        """Returns Physics object that is associated with the dm_env."""
        return self._env._physics.copy()

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
