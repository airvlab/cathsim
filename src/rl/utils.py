import os
import torch as th
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC, HerReplayBuffer

from imitation.algorithms import bc
from stable_baselines3.common import policies
from cathsim.utils import make_gym_env


class CnnPolicy(policies.ActorCriticCnnPolicy):
    """
    A CNN policy for behavioral clonning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "bc": bc,
}

RESULTS_PATH = Path(__file__).parent.parent / "results"
EXPERIMENT_PATH = RESULTS_PATH / "experiments"


def make_experiment(experiment_path: Path = None, base_path: Path = None) -> tuple:
    """Creates the maths for an experiment

    :param experiment_path: Path:  (Default value = None)
    :param base_path: Path:  (Default value = None)

    """
    assert experiment_path, "experiment_path must be specified"
    base_path = base_path or EXPERIMENT_PATH
    experiment_path = base_path / experiment_path
    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def train(
    algo: str,
    phantom: str = "phantom3",
    target: str = "bca",
    config_name: str = "test",
    base_path: Path = RESULTS_PATH,
    n_runs: int = 4,
    time_steps: int = 500_000,
    evaluate: bool = False,
    device: str = None,
    n_envs: int = None,
    vec_env: bool = True,
    config: dict = {},
    **kwargs,
) -> None:
    """Starts the training for an algorithm

    :param algo: str:
    :param experiment: str:
    :param experiment_path: Path:  (Default value = None)
    :param n_runs: int:  (Default value = 4)
    :param time_steps: int:  (Default value = 500_000)
    :param evaluate: bool:  (Default value = False)
    :param device: str:  (Default value = None)
    :param n_envs: int:  (Default value = None)
    :param vec_env: bool:  (Default value = True)
    :param config: dict:  (Default value = {})
    :param **kwargs:

    """
    from rl.evaluation import evaluate_policy

    algo_kwargs = config.get("algo_kwargs", {})

    if not device:
        device = "cuda" if th.cuda.is_available() else "cpu"
    n_envs = n_envs or os.cpu_count() // 2

    assert algo in ALGOS.keys(), f"algo must be one of {ALGOS.keys()}"
    assert n_runs > 0, "n_runs must be greater than 0"
    assert time_steps > 0, "time_steps must be greater than 0"
    assert device in ["cpu", "cuda"], "device must be one of [cpu, cuda]"
    assert n_envs > 0, "n_envs must be greater than 0"

    experiment_path = Path(f"{phantom}/{target}/{config_name}")
    model_path, log_path, eval_path = make_experiment(
        experiment_path, base_path=base_path
    )

    n_envs = n_envs or os.cpu_count() // 2

    env = make_gym_env(n_envs=n_envs, config=config)

    for seed in range(n_runs):
        if (model_path / f"{algo}_{seed}.zip").exists():
            print(f"Model {algo} {seed} already exists, skipping")
            pass
        else:
            for key, value in algo_kwargs.items():
                __import__("pprint").pprint(f"{key}: {value}")
            model = ALGOS[algo](
                env=env,
                device=device,
                verbose=1,
                tensorboard_log=log_path,
                **algo_kwargs,
            )

            model.learn(
                total_timesteps=time_steps,
                log_interval=10,
                tb_log_name=f"{algo}_{seed}",
                progress_bar=True,
                reset_num_timesteps=False,
            )

            model.save(model_path / f"{algo}_{seed}.zip")

            if evaluate:
                results = evaluate_policy(model, env, 10)
                np.savez_compressed(eval_path / f"{algo}_{seed}", **results)
            th.cuda.empty_cache()


def get_config(config_name: str) -> dict:
    """Parses a configuration file

    :param config_name: str:

    """
    import yaml
    from mergedeep import merge
    from rl.custom_extractor import CustomExtractor

    configs_path = Path(__file__).parent / "config"
    main_config_path = configs_path / "main.yaml"
    config_path = configs_path / (config_name + ".yaml")
    main_config = yaml.safe_load(open(main_config_path, "r"))
    config = yaml.safe_load(open(config_path, "r"))
    config = merge(main_config, config)

    policy_kwargs = main_config["algo_kwargs"].get("policy_kwargs", {})
    feature_extractor_class = policy_kwargs.get("features_extractor_class", None)
    if feature_extractor_class == "CustomExtractor":
        main_config["algo_kwargs"]["policy_kwargs"][
            "features_extractor_class"
        ] = CustomExtractor

    if main_config["algo_kwargs"].get("replay_buffer_class", None) == "HerReplayBuffer":
        main_config["algo_kwargs"]["replay_buffer_class"] = HerReplayBuffer
    return main_config
