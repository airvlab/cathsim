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
        channel_first: bool = False,
        time_limit: float = None,  # in seconds
    ):
        model_path = Path(__file__).parent.parent / "components/scene.xml"
        self.xml_path = model_path.resolve().as_posix()

        self.image_size = image_size
        self.image_fn = image_fn
        self.image_n_channels = image_n_channels
        self.channel_first = channel_first

        self._delta: float = 0.004
        self._success_reward: float = 10.0

        self._use_relative_position = True
        self._translation_step = translation_step
        self._rotation_step = math.radians(rotation_step)

        self.spec, self.model, self.data = self._initialize_simulation()

        # used to reset the model
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_guidewire_pos = self.model.body("guidewire").pos.copy()

        self.frame_skip = 59
        self.time_limit = time_limit

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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _initialize_simulation(
        self,
    ) -> Tuple["mujoco.MjModel", "mujoco.MjData"]:
        spec = mujoco.MjSpec()
        spec.from_file(self.xml_path)
        model = spec.compile()

        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.image_size
        model.vis.global_.offheight = self.image_size
        data = mujoco.MjData(model)
        return spec, model, data

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

    def do_simulation(self, ctrl, n_frames, tolerance=1e-3, max_iters=1000) -> None:
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")

        self._step_mujoco_simulation(ctrl, n_frames)

        # Wait for the joint to reach the desired position within the specified tolerance
        for i in range(max_iters):
            # Step the simulation
            self._step_mujoco_simulation(ctrl, n_frames)

            # Check if the joint positions are within the tolerance
            current_j_pos = self._current_j_pos
            if all(np.abs(np.array(current_j_pos) - np.array(ctrl)) <= tolerance):
                print(f"Joint position converged within {i} iterations.")
                break
        else:
            print(f"Warning: Joint position did not converge within {max_iters} iterations.")

    @property
    def _current_j_pos(self):
        slider_joint = self.data.joint("slider").qpos[0]
        rotator_joint = self.data.joint("rotator").qpos[0]
        return [slider_joint, rotator_joint]

    def _relative_to_global_action(self, action):
        translation = action[0] * self._translation_step
        rotation = action[1] * self._rotation_step
        action = np.array([translation, rotation])
        action = self._current_j_pos + action
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
        info["action"] = [round(action[0], 4), round(math.degrees(action[1]))]

        reward = self.compute_reward(achieved_goal=head_position, desired_goal=target_position, info=info)

        terminated = self.compute_terminated(achieved_goal=head_position, desired_goal=target_position, info=info)
        truncated = info["time"] >= self.time_limit if self.time_limit is not None else False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

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

    def make_movie(frames, fps):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

    env = gym.make("CathSim-v1", render_mode="rgb_array", image_size=480)
    # check_env(env.unwrapped, skip_render_check=True)
    # env = CathSim(render_mode="rgb_array", image_size=480)

    print(env.action_space)
    print(env.observation_space)

    for ep in range(4):
        done = False
        ob, info = env.reset()
        step = 0
        frames = []
        while not done:
            action = env.action_space.sample()
            action[0] = 1
            action[1] = -1
            ob, reward, terminated, truncated, info = env.step(action)
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)
            print(f"Step {step:03d} ({info['time']:.1f}): Force {info['total_force']:.3f} Action {info['action']}")
            cv2.imshow("Top Camera", img)
            cv2.waitKey(1)
            done = terminated or truncated
            step += 1
        fps = len(frames) / info["time"]
        make_movie(frames, fps)
        exit()
        print("Time elapsed: ", info["time"])
        cv2.destroyAllWindows()
