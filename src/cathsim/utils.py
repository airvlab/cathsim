"""
Utilities module for the environment.
"""
from cathsim.wrappers import (
    DMEnvToGymWrapper,
    GoalEnvWrapper,
    MultiInputImageWrapper,
    Dict2Array,
)
from gym import wrappers


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt


import gym
from dm_control.viewer.application import Application
from dm_control import composer


def normalize_rgba(rgba: list) -> list:
    new_rgba = [c / 255.0 for c in rgba]
    new_rgba[-1] = rgba[-1]
    return new_rgba


def point2pixel(
    point: np.ndarray, camera_kwargs: dict = dict(image_size=80)
) -> np.ndarray:
    """Transforms from world coordinates to pixel coordinates.

    Args:
      point: np.ndarray: the point to be transformed.
      camera_kwargs: dict:  (Default value = dict(image_size=80))

    Returns:
        np.ndarray : the pixel coordinates of the point.
    """
    camera_matrix = create_camera_matrix(**camera_kwargs)
    x, y, z = point
    xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

    return np.array([round(xs / s), round(ys / s)]).astype(np.int32)


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


def create_camera_matrix(
    image_size: int,
    pos=np.array([-0.03, 0.125, 0.15]),
    euler=np.array([0, 0, 0]),
    fov=45,
) -> np.ndarray:
    """
    Generate a camera matrix given the parameters

    Args:
      image_size: int:
      pos:  (Default value = np.array([-0.03, 0.125, 0.15]):
      euler:  (Default value = np.array([0, 0, 0]):
      fov:  (Default value = 45)

    Returns:
        np.ndarray: The camera matrix

    """

    def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        Make a rotation matrix from euler angles

        Args:
            euler_angles: np.ndarray: The euler angles

        Returns:
            np.ndarray: The rotation matrix

        """
        rx, ry, rz = np.deg2rad(euler_angles)

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )

        Ry = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )

        Rz = np.array(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
        )

        R = Rz @ Ry @ Rx
        return R

    # Intrinsic Parameters
    focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2)) * image_size / 2.0
    focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    image = np.eye(3)
    image[0, 2] = (image_size - 1) / 2.0
    image[1, 2] = (image_size - 1) / 2.0

    # Extrinsic Parameters
    rotation_matrix = euler_to_rotation_matrix(euler)
    R = np.eye(4)
    R[0:3, 0:3] = rotation_matrix
    T = np.eye(4)
    T[0:3, 3] = -pos

    # Camera Matrix
    camera_matrix = image @ focal @ R @ T
    return camera_matrix


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


def plot_w_mesh(mesh, points: np.ndarray, **kwargs):
    """
    Plot a mesh with points

    Args:
        mesh: The mesh to plot
        points: The points to plot
        **kwargs: The keyword arguments to pass to the scatter function

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    ax.scatter(points[:, 0], points[:, 0], points[:, 0], **kwargs)
    plt.show()


def make_dm_env(
    phantom: str = "phantom3",
    target: str = "bca",
    use_pixels: bool = False,
    dense_reward: bool = True,
    success_reward: float = 10.0,
    delta: float = 0.004,
    use_segment: bool = False,
    image_size: int = 80,
    **kwargs,
) -> composer.Environment:
    """Makes a dm_control environment given a configuration.

    Args:
      phantom: str:  (Default value = "phantom3") The phantom to use
      target: str:  (Default value = "bca") The target to use
      use_pixels: bool:  (Default value = False) Whether or not to use pixels
      dense_reward: bool:  (Default value = True) Whether or not to use dense reward
      success_reward: float:  (Default value = 10.0) The reward for success
      delta: float:  (Default value = 0.004) The delta for the reward
      use_segment: bool:  (Default value = False) Whether or not to use the segment image
      image_size: int:  (Default value = 80) The size of the image
      **kwargs:

    Returns:
        composer.Environment: The environment

    """

    from cathsim import Phantom, Tip, Guidewire, Navigate

    phantom = phantom + ".xml"

    phantom = Phantom(phantom)
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        dense_reward=dense_reward,
        success_reward=success_reward,
        delta=delta,
        use_pixels=use_pixels,
        use_segment=use_segment,
        image_size=image_size,
        target=target,
        **kwargs,
    )
    env = composer.Environment(
        task=task,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    return env


def make_gym_env(
    config: dict = {}, n_envs: int = 1, monitor_wrapper: bool = True
) -> gym.Env:
    """Makes a gym environment given a configuration. This is a wrapper for the creation of environment and basic wrappers

    Args:
        config: dict:  (Default value = {}) The configuration dictionary
        n_envs: int:  (Default value = 1) The number of environments to create
        monitor_wrapper: bool:  (Default value = True) Whether or not to use the monitor wrapper

    Returns:
        gym.Env: The environment

    """

    def _create_env() -> gym.Env:
        """Create and return environment based on config. This is a wrapper for the creation of environment and basic wrappers

        Returns:
            gym.Env: The environment

        """
        # Extract parameters from the config dictionary.
        wrapper_kwargs = config.get("wrapper_kwargs", {})
        env_kwargs = config.get("env_kwargs", {})
        task_kwargs = config.get("task_kwargs", {})

        # Environment creation and basic wrapping
        env = make_dm_env(**task_kwargs)
        env = DMEnvToGymWrapper(env=env, env_kwargs=env_kwargs)

        # If goal_env is set to True then the goal is used to determine the desired goal.
        if wrapper_kwargs.get("goal_env", False):
            filter_keys = wrapper_kwargs.get("use_obs", []) + [
                "achieved_goal",
                "desired_goal",
            ]
            env = GoalEnvWrapper(env=env)
        else:
            filter_keys = wrapper_kwargs.get("use_obs", [])

        # FilterObservation for the filter_keys.
        if filter_keys:
            env = wrappers.FilterObservation(env, filter_keys=filter_keys)

        env = wrappers.TimeLimit(
            env, max_episode_steps=wrapper_kwargs.get("time_limit", 300)
        )

        if wrapper_kwargs.get("flatten_obs", False):
            env = wrappers.FlattenObservation(env)

        if task_kwargs.get("use_pixels", False):
            env = MultiInputImageWrapper(
                env,
                grayscale=wrapper_kwargs.get("grayscale", False),
                image_key=wrapper_kwargs.get("image_key", "pixels"),
                keep_dim=wrapper_kwargs.get("keep_dim", True),
                channel_first=wrapper_kwargs.get("channel_first", False),
            )

        # If the observation dict has a single key, flatten the observation.
        if wrapper_kwargs.get("dict2array", False):
            assert (
                len(env.observation_space.spaces) == 1
            ), "Only one observation is allowed."
            env = Dict2Array(env)

        # NormalizeObservation if normalize_obs is True.
        if wrapper_kwargs.get("normalize_obs", False):
            env = wrappers.NormalizeObservation(env)

        if wrapper_kwargs.get("frame_stack", 1) > 1:
            env = wrappers.FrameStack(env, wrapper_kwargs["frame_stack"])

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
        from rl.data import Trajectory

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
        from rl.data import Trajectory

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
