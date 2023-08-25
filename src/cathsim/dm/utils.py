"""
Utilities module for the environment.
"""
import numpy as np
import yaml
from gymnasium import wrappers
import gymnasium as gym
from cathsim.gym.wrappers import (
    GoalEnvWrapper,
    MultiInputImageWrapper,
    SingleDict2Array,
)


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from pathlib import Path
from dm_control.viewer.application import Application
from toolz.dicttoolz import itemmap


def flatten_dict(d: dict, parent_key: str = None) -> dict:
    acc = {}
    for k, v in d.items():
        if parent_key:
            k = parent_key + "-" + k
        if isinstance(v, dict):
            acc = acc | flatten_dict(v, k)
        else:
            acc[k] = v
    return acc


def expand_dict(xd: dict, yd: dict) -> dict:
    zd = xd.copy()
    for k, v in yd.items():
        if k not in xd:
            zd[k] = [v]
        elif isinstance(v, dict) and isinstance(xd[k], dict):
            zd[k] = expand_dict(xd[k], v)
        else:
            zd[k] = xd[k] + [v]
    return zd


def map_val(g: callable, d: dict):
    def f(item):
        k, v = item
        if isinstance(v, dict):
            return (k, itemmap(f, v))
        else:
            return (k, g(v))

    return itemmap(f, d)


def normalize_rgba(rgba: list) -> list:
    new_rgba = [c / 255.0 for c in rgba]
    new_rgba[-1] = rgba[-1]
    return new_rgba


def filter_mask(segment_image: np.ndarray):
    """
    Convert the segment image to a mask

    Args:
      segment_image: np.ndarray: The segment image

    Returns:
        np.ndarray: The mask
    """
    geom_ids = segment_image[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    segment_image = 255 * geom_ids
    return segment_image


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the Euclidean distance

    Args:
      a: np.ndarray: The first point
      b: np.ndarray: The second point

    Returns:
        np.ndarray: The distance between the two points
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


def get_env_config(config_path: Path = None) -> dict:
    """Read and parse `env_config` yaml file.

    Args:
      config_path: Path:  (Default value = None)

    Returns:
        dict: The configuration dictionary

    """
    if config_path is None:
        config_path = Path(__file__).parent / "env_config.yaml"
    with open(config_path, "r") as f:
        env_config = yaml.safe_load(f)
    return env_config


WRAPPERS = {
    "GoalEnvWrapper": (GoalEnvWrapper, {}),
    "FilterObservation": (
        wrappers.FilterObservation,
        {"filter_keys": lambda cfg: cfg["wrapper_kwargs"].get("use_obs", [])},
    ),
    "TimeLimit": (
        wrappers.TimeLimit,
        {"max_episode_steps": lambda cfg: cfg["wrapper_kwargs"].get("time_limit", 300)},
    ),
    "FlattenObservation": (wrappers.FlattenObservation, {}),
    "MultiInputImageWrapper": (
        MultiInputImageWrapper,
        {
            "grayscale": lambda cfg: cfg["wrapper_kwargs"].get("grayscale", True),
            "image_key": lambda cfg: cfg["wrapper_kwargs"].get("image_key", "pixels"),
            "keep_dim": lambda cfg: cfg["wrapper_kwargs"].get("keep_dim", True),
            "channel_first": lambda cfg: cfg["wrapper_kwargs"].get(
                "channel_first", False
            ),
        },
    ),
    "SingleDict2Array": (SingleDict2Array, {}),
    "NormalizeObservation": (wrappers.NormalizeObservation, {}),
    "FrameStack": (
        wrappers.FrameStack,
        {"num_stack": lambda cfg: cfg["wrapper_kwargs"].get("frame_stack", 1)},
    ),
}


def make_gym_env(
    config: dict = {}, n_envs: int = 1, monitor_wrapper: bool = True, wrappers=WRAPPERS
) -> gym.Env:
    def _create_env() -> gym.Env:
        from gymnasium import make

        env = make("cathsim/CathSim-v0", **wrappers.get("task_kwargs", {}))

        for wrap in wrappers:
            if wrap["condition"](config):
                dynamic_kwargs = {
                    k: (v(config) if callable(v) else v)
                    for k, v in wrap["kwargs"].items()
                }
                env = wrap["wrapper"](env, **dynamic_kwargs)

        return env

    if n_envs > 1:
        envs = [_create_env for _ in range(n_envs)]
        env = SubprocVecEnv(envs)
    else:
        env = _create_env()

    if monitor_wrapper:
        env = Monitor(env) if n_envs == 1 else VecMonitor(env)

    return env


class Application(Application):
    """Augmented interface that allows keyboard control"""

    def __init__(
        self,
        title,
        width,
        height,
        save_trajectories: bool = False,
        phantom: str = "phantom3",
        target: str = "bca",
        experiment_name: str = None,
        base_path: Path = "results-test",
        resume: bool = True,
    ):
        """
        Initialize the Application.

        Args:
            title: Title of the window.
            width: Width of the window in pixels. Must be greater than 0.
            height: Height of the window in pixels. Must be greater than 0.
            save_trajectories: If True ( default ) trajectories will be saved to a file for use with : py : meth : ` open `.
            phantom: Specifies the type of phantom to use.
            target: Specifies the target to use.
            experiment_name: If specified the experiment will be used as base for the path to the file.
            base_path: The base path to save the data to.
            resume: Resume the experiment after this call. Defaults to
        """
        super().__init__(title, width, height)
        from dm_control.viewer import user_input
        from cathsim.rl.data import Trajectory

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)

        self.save_trajectories = save_trajectories
        if self.save_trajectories:
            self.trajectory = Trajectory()
            self.path = Path.cwd() / base_path / experiment_name / phantom / target
            self.path.mkdir(parents=True, exist_ok=True)

        self.null_action = np.zeros(2)
        self._step = 0
        self._episode = 0
        if resume and self.save_trajectories:
            self._episode = sorted(self.path.iterdir())[-1] + 1
        self._policy = None

    def _initialize_episode(self):
        """ """
        from cathsim.rl.data import Trajectory

        if self.save_trajectories:
            self.trajectory.save(self.path / str(self._episode))
            self.trajectory = Trajectory()
            self._step = 0
            self._episode += 1
            print(f"Episode {self._episode:02} finished")
        self._restart_runtime()

    def perform_action(self):
        """ """
        time_step = self._runtime._time_step
        if not time_step.last():
            self._advance_simulation()
            if self.save_trajectories:
                print(f"step {self._step:03}")
                action = self._runtime._last_action
                time_step.observation["action"] = action
                self.trajectory.add_transition(**time_step.observation)
                self._step += 1
        else:
            self._initialize_episode()

    def _move_forward(self):
        """ """
        self._runtime._default_action = [1, 0]
        self.perform_action()

    def _move_back(self):
        """ """
        self._runtime._default_action = [-1, 0]
        self.perform_action()

    def _move_left(self):
        """ """
        self._runtime._default_action = [0, -1]
        self.perform_action()

    def _move_right(self):
        """ """
        self._runtime._default_action = [0, 1]
        self.perform_action()


def launch(
    environment_loader,
    policy=None,
    title="Explorer",
    width=1024,
    height=768,
    save_trajectories: bool = False,
    **kwargs,
):
    """Launches the environment. This is to be used for manual control of for visualizing a dm_env policy.

    Args:
      environment_loader: The environment to use.
      policy: The policy to use. Defaults to None which means no policy is used.
      title: The title of the application.
      width: The width of the application in pixels.
      height: The height of the application in pixels.
      save_trajectories: Whether or not to save trajectories
      save_trajectories: bool: (Default value = False) Whether or not to save trajectories
      **kwargs:

    """

    app = Application(
        title=title,
        width=width,
        height=height,
        save_trajectories=save_trajectories,
        **kwargs,
    )
    app.launch(
        environment_loader=environment_loader,
        policy=policy,
    )
