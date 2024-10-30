from typing import Any, Dict, Optional

import gymnasium as gym


def apply_filter_observation(env: gym.Env, filter_keys: Optional[list]) -> gym.Env:
    if filter_keys:
        from gymnasium import wrappers

        env = wrappers.FilterObservation(env, filter_keys=filter_keys)
    return env


def apply_multi_input_image_wrapper(env: gym.Env, options: Dict[str, Any]) -> gym.Env:
    if options.get("use_pixels", False):
        from cathsim.gym.wrappers import MultiInputImageWrapper

        env = MultiInputImageWrapper(
            env,
            grayscale=options.get("grayscale", False),
            image_key=options.get("image_key", "pixels"),
            keep_dim=options.get("keep_dim", True),
            channel_first=options.get("channel_first", False),
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
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    wrapper_kwargs = config.wrapper_kwargs or {}
    task_kwargs = config.task_kwargs or {}

    def _create_env() -> gym.Env:
        from cathsim.gym.envs import CathSim

        env = gym.make("cathsim/CathSim-v0", **task_kwargs)

        env = apply_filter_observation(
            env, filter_keys=wrapper_kwargs.get("use_obs", [])
        )

        env = apply_multi_input_image_wrapper(
            env,
            options={
                "grayscale": wrapper_kwargs.get("grayscale", True),
                "image_key": wrapper_kwargs.get("image_key", "pixels"),
                "keep_dim": wrapper_kwargs.get("keep_dim", True),
                "channel_first": wrapper_kwargs.get("channel_first", False),
            },
        )

        return env

    if n_envs > 1:
        envs = [_create_env for _ in range(n_envs)]
        env = SubprocVecEnv(envs)
    else:
        env = _create_env()

    if monitor_wrapper:
        env = Monitor(env) if n_envs == 1 else VecMonitor(env)

    return env


if __name__ == "__main__":
    from cathsim.rl import Config

    config = Config()
    env = make_gym_env(config, n_envs=1, monitor_wrapper=True)
    print(env)
