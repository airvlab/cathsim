import os
import torch as th
from pathlib import Path


from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}

RESULTS_PATH = Path.cwd() / "results"


def generate_experiment_paths(experiment_path: Path = None) -> tuple:
    if experiment_path.is_absolute() is False:
        experiment_path = RESULTS_PATH / experiment_path

    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for directory_path in [experiment_path, model_path, log_path, eval_path]:
        directory_path.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def load_sb3_model(path: Path, config_name: str = None) -> BaseAlgorithm:
    config = Config(config_name)
    algo_kwargs = config.get("algo_kwargs", {})

    model = SAC.load(
        path,
        custom_objects={"policy_kwargs": algo_kwargs.get("policy_kwargs", {})},
    )
    return model


if __name__ == "__main__":
    from cathsim.rl import Config, make_gym_env
    from pprint import pprint
    import stable_baselines3

    config = Config()
    # print(config)
    env = make_gym_env(config=config, n_envs=1)
    pprint(config.algo_kwargs)
