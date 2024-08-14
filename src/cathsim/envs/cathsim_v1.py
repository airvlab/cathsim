import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import Renderer
from numpy.typing import NDArray

DEFAULT_SIZE = 480


class CathSim(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
    }

    def __init__(
        self,
        render_mode: str = "rgb_array",
        image_size: int = DEFAULT_SIZE,
        image_fn: callable = None,
        translation_step: float = 0.001,  # in meters
        rotation_step: float = 5,  # in degrees
        image_n_channels: int = 3,
    ):
        model_path = Path(__file__).parent.parent / "components/scene.xml"
        self.xml_path = model_path.resolve().as_posix()

        self.image_size = image_size
        self.image_fn = image_fn
        self.image_n_channels = image_n_channels

        self._delta: float = 0.004
        self._success_reward: float = 10.0

        self._use_relative_position = True
        self._translation_step = translation_step
        self._rotation_step = math.radians(rotation_step)

        self.model, self.data = self._initialize_simulation()

        # used to reset the model
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_guidewire_pos = self.model.body("guidewire").pos.copy()

        self.frame_skip = 7

        assert (
            render_mode in self.metadata["render_modes"]
        ), f"Invalid render mode {render_mode}, must be one of {self.metadata['render_modes']}"
        self.render_mode = render_mode

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        self._set_action_space()
        self._set_observation_space()

        self._target_position = None

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
        )

        self.translation_step = 0.01
        self.bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)

    def _set_observation_space(self):
        image_space = spaces.Box(
            0,
            255,
            shape=(self.image_size, self.image_size, self.image_n_channels),
            dtype=np.uint8,
        )

        obs_space = spaces.Dict(
            dict(
                pixels=image_space,
            )
        )
        self.observation_space = obs_space

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self._use_relative_position:
            self.action_space = spaces.Box(
                -1.0, 1.0, shape=(len(low),), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _initialize_simulation(
        self,
    ) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.image_size
        model.vis.global_.offheight = self.image_size
        data = mujoco.MjData(model)
        return model, data

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def render(self):
        """
        Render a frame from the MuJoCo simulation as specified by the render_mode.
        """
        return self.mujoco_renderer.render(self.render_mode, camera_name="top")

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    @property
    def _current_control(self):
        return self.data.ctrl

    def _relative_to_global_action(self, action):
        translation = action[0] * self._translation_step
        rotation = action[1] * self._rotation_step
        action = np.array([translation, rotation])
        action = self._current_control + action
        action = np.clip(action, self.bounds[:, 0], self.bounds[:, 1])
        return action

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        action = self._relative_to_global_action(action)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        head_position = self._tip_position
        target_position = self._target_position

        info = self._get_info()
        info["action"] = action

        reward = self.compute_reward(
            achieved_goal=head_position, desired_goal=target_position, info=info
        )

        terminated = self.compute_terminated(
            achieved_goal=head_position, desired_goal=target_position, info=info
        )

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        if distance < self._delta:
            return self._success_reward
        return -distance

    def compute_terminated(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self._delta

    def reset_model(self) -> NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel

        # guidewire_pos = self.init_guidewire_pos + np.random.uniform(
        #     -0.001, 0.001, size=3
        # )
        #
        # self.model.body("guidewire").pos = guidewire_pos

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def _get_obs(self):
        top_img = self.mujoco_renderer.render("rgb_array", camera_name="top")
        if self.image_fn:
            top_img = self.image_fn(top_img)
        return {"pixels": top_img}

    def _get_info(self):
        return dict(
            head_pos=self._tip_position.copy(),
            target_pos=self.target_position.copy(),
            time=self.data.time,
            total_force=self._total_force.copy(),
        )

    @property
    def _tip_position(self):
        tip_pos = self.data.body("tipB_last").xpos
        return tip_pos

    @property
    def target_position(self):
        if self._target_position is None:
            self._target_position = self.data.site("bca").xpos
        return self._target_position

    @target_position.setter
    def target_position(self, value):
        self._target_position = value

    @property
    def _total_force(self):
        forces = self.data.qfrc_constraint[0:3]
        return np.linalg.norm(forces)


if __name__ == "__main__":
    from pathlib import Path

    import cv2
    from gymnasium.utils.env_checker import check_env

    import cathsim

    env = gym.make("CathSim-v1", render_mode="rgb_array", image_size=480)
    # check_env(env.unwrapped, skip_render_check=True)
    # env = CathSim(render_mode="rgb_array", image_size=480)

    print(env.action_space)
    print(env.observation_space)

    for ep in range(4):
        done = False
        ob, info = env.reset()
        while not done:
            action = env.action_space.sample()
            action[0] = 1
            action[1] = 1
            ob, reward, terminated, truncated, info = env.step(action)
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Top Camera", img)
            cv2.waitKey(1)
            done = terminated or truncated
        print("Time elapsed: ", info["time"])
        cv2.destroyAllWindows()
