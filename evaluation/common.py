"""Common functions for experiments."""
from pathlib import Path
import numpy as np
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from imitation.algorithms import bc

from rl.utils import get_config
from cathsim.utils import make_gym_env

import pandas as pd
import tqdm
from typing import Union, List


ALGOS = {
    "sac": SAC,
    "bc": bc,
}

RESULTS_PATH = Path.cwd() / "results" / "nips2023-Jun-23"
EXPERIMENT_PATH = RESULTS_PATH / "experiments"
EVALUATION_PATH = RESULTS_PATH / "evaluation"

PHANTOMS = ["phantom3"]
TARGETS = ["bca", "lcca"]
CONFIGS = ["full", "internal_pixels", "internal", "pixels", "pixel_mask"]


def make_experiment(experiment_path: Path = None) -> tuple:
    assert experiment_path, "experiment_path must be specified"
    experiment_path = EXPERIMENT_PATH / experiment_path
    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def get_paths(path: Path) -> Union[List[Path], List[Path], List[Path]]:
    model_path, log_path, eval_path = make_experiment(path)
    models_path = model_path.glob("*.zip")
    models_path = sorted(models_path, key=lambda x: x.stem.split("_")[-1])
    logs_paths = log_path.rglob("*tfevents*")
    logs_paths = sorted(logs_paths, key=lambda x: x.stem.split(".")[3])
    eval_paths = eval_path.rglob("*.npz")
    eval_paths = sorted(eval_paths, key=lambda x: x.stem.split("_")[-1])
    return models_path, logs_paths, eval_paths


def parse_tb_log(path: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tag = "rollout/ep_len_mean"
    acc = EventAccumulator(path.as_posix())
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        print("Tag not found")
        return
    df = pd.DataFrame(acc.Scalars(tag))
    df = df.drop(columns=["wall_time"]).groupby("step").mean()
    return df


def get_tb_logs(
    path: Path,
    n_interpolations: int = 30,
    max_timesteps: int = 600_000,
    length_threshold: int = 150,
):
    _, log_paths, _ = get_paths(path)
    logs = []
    for log in log_paths:
        tensorboard_log = parse_tb_log(log)
        if tensorboard_log is None:
            continue
        assert len(tensorboard_log) != 0, f"Log {log} is empty."
        if len(tensorboard_log) < length_threshold:
            continue
        tensorboard_log.reindex(
            np.arange(0, max_timesteps, max_timesteps / n_interpolations),
            method="nearest",
        )
        logs.append(tensorboard_log)
    if len(logs) > 0:
        mean = pd.concat(logs, axis=0).groupby(level=0).mean().squeeze()
        stdev = pd.concat(logs, axis=0).groupby(level=0).std().squeeze()
        return mean, stdev
    else:
        return None


def evaluate_models(
    experiments_path: Path = None,
    n_episodes=10,
    phantom_name: str = None,
    target_name: str = None,
    algorithm_name: str = None,
):
    """
    Evaluate the performance of all the models in the experiments directory.

    :param experiments_path Path: The path to the experiments directory.
    :param n_episodes int: The number of episodes to evaluate the policy for.
    """
    if not experiments_path:
        experiments_path = EXPERIMENT_PATH

    phantoms = [
        phantom
        for phantom in experiments_path.iterdir()
        if (phantom.is_dir() and (phantom_name is None or phantom.name == phantom_name))
    ]
    for phantom in phantoms:
        targets = [
            target
            for target in phantom.iterdir()
            if (target.is_dir() and (target_name is None or target.name == target_name))
        ]
        for target in targets:
            algorithms = [
                algorithm
                for algorithm in target.iterdir()
                if ((algorithm_name is None or algorithm.name == algorithm_name))
                and algorithm.is_dir()
            ]
            for algorithm in algorithms:
                evaluate_model(algorithm, n_episodes)


def evaluate_model(algorithm_path: Path, n_episodes=10):
    models_path, _, eval_path = make_experiment(algorithm_path)
    models_paths = models_path.glob("*.zip")
    models_paths = sorted(models_paths, key=lambda x: x.stem.split("_")[-1])

    for model_path in models_paths:
        model_name = model_path.stem
        if (eval_path / (model_name + ".npz")).exists():
            continue
        print(f"Evaluating {model_name} in {algorithm_path} for {n_episodes} episodes.")

        config = get_config(algorithm_path.stem)
        config["task_kwargs"]["phantom"] = algorithm_path.parent.parent.stem
        config["task_kwargs"]["target"] = algorithm_path.parent.stem

        if "bc" in algorithm_path.stem:
            config["wrapper_kwargs"]["channel_first"] = True

        env = make_gym_env(config)

        if "bc" in algorithm_path.stem:
            from scratch.bc.custom_networks import CnnPolicy
            from cathsim.wrappers import Dict2Array
            import torch as th

            env = Dict2Array(env)

            model = CnnPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda _: th.finfo(th.float32).max,
            ).load(models_path / "bc")
        else:
            model = SAC.load(model_path)
        evaluation_data = evaluate_policy(model, env, n_episodes=n_episodes)
        np.savez_compressed(eval_path / f"{model_name}.npz", **evaluation_data)


def evaluate_policy(model: BaseAlgorithm, env: gym.Env, n_episodes: int = 10) -> dict:
    evaluation_data = {}
    for episode in tqdm(range(n_episodes)):
        observation = env.reset()
        done = False
        head_positions = []
        forces = []
        head_pos = env.head_pos.copy()
        head_positions.append(head_pos)
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            head_pos_ = info["head_pos"]
            forces.append(info["forces"])
            head_positions.append(head_pos_)
            head_pos = head_pos_
        evaluation_data[str(episode)] = dict(
            forces=np.array(forces),
            head_positions=np.array(head_positions),
        )
    return evaluation_data


if __name__ == "__main__":
    for phantom in PHANTOMS:
        for target in TARGETS:
            for config in CONFIGS:
                path = Path(f"{phantom}/{target}/{config}")
                get_paths(path)
                evaluate_model(path)
