from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

DEFAULT_SIZE = 480


def transform_inage(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


class CathSim(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):
        model_path = Path(__file__).parent.parent / "components/scene.xml"
        self.xml_path = model_path.resolve().as_posix()

        self.width = width
        self.height = height

        self._delta: float = 0.004
        self._success_reward: float = 10.0
        self._use_relative_position = True

        self.model, self.data = self._initialize_simulation()

        # used to reset the model
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._translation_step = 0.1
        self._rotation_step = 0.1

        self.frame_skip = frame_skip

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
        ], self.metadata["render_modes"]
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self._set_action_space()
        self._set_observation_space()
        self.target_position = None

        self.render_mode = render_mode

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
        )

        self.translation_step = 0.01
        self.bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)

    def _set_observation_space(self):
        image_space = spaces.Box(
            0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
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
                low=np.array([-self._translation_step, -self._rotation_step]),
                high=np.array([self._translation_step, self._rotation_step]),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _initialize_simulation(
        self,
    ) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def set_state(self, qpos, qvel):
        """Set the joints position qpos and velocity qvel of the model.

        Note: `qpos` and `qvel` is not the full physics state for all mujoco models/environments https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
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
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close rendering contexts processes."""
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        """Return the cartesian position of a body frame."""
        return self.data.body(body_name).xpos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()

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

    def step(
        self, action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        if self._use_relative_position:
            action = self._current_control + action
            action = np.clip(action, self.bounds[:, 0], self.bounds[:, 1])
            print("Internal action: ", action)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        head_position = self.head_position
        target_position = self.target_position

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
        # TODO: replace the hardcoded value
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self._delta

    def reset_model(self) -> NDArray[np.float64]:
        # can add noise here
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self) -> Dict[str, float]:
        return self._get_info()

    def _get_obs(self):
        top_img = self.mujoco_renderer.render("rgb_array", camera_name="top")
        top_img = transform_inage(top_img)
        return {"pixels": top_img}

    def _get_info(self):
        return dict(
            head_pos=self.head_position.copy(),
            target_pos=self.target_position.copy(),
        )

    @property
    def head_position(self):
        head_pos = self.data.body("tipB_last").xpos
        return head_pos

    @property
    def target_position(self):
        if self._target_position is None:
            self._target_position = self.data.site("bca").xpos
        return self._target_position

    @target_position.setter
    def target_position(self, value):
        self._target_position = value


if __name__ == "__main__":
    from pathlib import Path

    import cv2

    cathsim = CathSim(
        render_mode="rgb_array",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
    )
    print(cathsim.action_space)
    print(cathsim.observation_space)

    ob, info = cathsim.reset()
    for _ in range(1000):
        # action = cathsim.action_space.sample()
        action = [0.1, 0]
        ob, reward, terminated, _, info = cathsim.step(action)
        print(f"Reward: {reward}, Info: {info}")
        cv2.imshow("Top Camera", ob["pixels"])
        cv2.waitKey(1)
