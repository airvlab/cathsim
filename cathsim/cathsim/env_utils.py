import numpy as np
from dm_control.viewer.application import Application
from dm_control import composer
import gym


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    return np.linalg.norm(a - b, axis=-1)


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
    """
    Makes a dm_control environment.

    :param dense_reward: If True, the reward is the negative distance to the target.
    :param success_reward: The reward when the target is reached.
    :param delta: The distance to the target to be considered a success.
    :param use_pixels: If True, an image of the scene is returned as part of the observation.
    :param use_segment: If True, the guidewire segmentation is returned as part of the observation.
    :param image_size: The size of the image to be returned or rendered.
    """
    from cathsim.cathsim import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer

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
    config: dict = dict(
        task_kwargs={},
        wrapper_kwargs={},
        render_kwargs={},
    )
) -> gym.Env:
    wrapper_kwargs = config["wrapper_kwargs"]
    env_kwargs = config["env_kwargs"]
    task_kwargs = config["task_kwargs"]

    gym_version = env_kwargs["gym_version"]
    if gym_version == "gym":
        from cathsim.wrappers import DMEnvToGymWrapper
        from gym import wrappers
    elif gym_version == "gymnasium":
        from wrapper_gymnasium import DMEnvToGymWrapper
        from gymnasium import wrappers

    max_episode_steps = wrapper_kwargs.get("time_limit", 300)
    filter_keys = wrapper_kwargs.get("use_obs", None)
    flatten_observation = wrapper_kwargs.get("flatten_obs", False)
    grayscale = wrapper_kwargs.get("grayscale", False)
    normalize_obs = wrapper_kwargs.get("normalize_obs", False)
    frame_stack = wrapper_kwargs.get("frame_stack", 1)
    use_pixels = task_kwargs.get("use_pixels", False)
    goal_env = wrapper_kwargs.get("goal_env", False)

    if goal_env:
        filter_keys = filter_keys + ["achieved_goal", "desired_goal"]

    env = make_dm_env(**task_kwargs)
    env = DMEnvToGymWrapper(env=env, env_kwargs=env_kwargs)

    if goal_env:
        from cathsim.wrappers import GoalEnvWrapper

        env = GoalEnvWrapper(env=env)

    env = wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if gym_version == "gymnasium":
        env = wrappers.EnvCompatibility(env, render_mode="rgb_array")

    if filter_keys:
        env = wrappers.FilterObservation(env, filter_keys=filter_keys)

    if flatten_observation:
        env = wrappers.FlattenObservation(env)

    if use_pixels:
        from cathsim.wrappers import MultiInputImageWrapper

        env = MultiInputImageWrapper(
            env,
            grayscale=grayscale,
            image_key=wrapper_kwargs.get("image_key", "pixels"),
            keep_dim=wrapper_kwargs.get("keep_dim", True),
            channel_first=wrapper_kwargs.get("channel_first", False),
        )

    if wrapper_kwargs.get("dict2array", False):
        assert (
            len(env.observation_space.spaces) == 1
        ), "Only one observation is allowed."
        from cathsim.wrappers import Dict2Array

        env = Dict2Array(env)

    if normalize_obs:
        env = wrappers.NormalizeObservation(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, frame_stack)

    return env


class Application(Application):
    def __init__(self, title, width, height):
        super().__init__(title, width, height)
        from dm_control.viewer import user_input

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)
        self.null_action = np.zeros(2)
        self._step = 0
        self._policy = None

    def _initialize_episode(self):
        self._restart_runtime()
        self._step = 0

    def perform_action(self):
        time_step = self._runtime._time_step
        physics = self._runtime._env.physics
        force = self._runtime._env.task.get_force(physics).round(2)
        print(f"step {self._step:03} | force {force}")
        if not time_step.last():
            self._advance_simulation()
            self._step += 1
        else:
            self._initialize_episode()

    def _move_forward(self):
        self._runtime._default_action = [1, 0]
        self.perform_action()

    def _move_back(self):
        self._runtime._default_action = [-1, 0]
        self.perform_action()

    def _move_left(self):
        self._runtime._default_action = [0, -1]
        self.perform_action()

    def _move_right(self):
        self._runtime._default_action = [0, 1]
        self.perform_action()


def launch(
    environment_loader,
    policy=None,
    title="Explorer",
    width=1024,
    height=768,
    trial_path=None,
):
    app = Application(title=title, width=width, height=height)
    app.launch(environment_loader=environment_loader, policy=policy)


if __name__ == "__main__":
    from rl.utils import get_config

    config = get_config("test")
    __import__("pprint").pprint(config)
    # env = make_env(config)
    env = make_dm_env()
    launch(env, policy=None)
