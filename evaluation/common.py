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
from typing import Union, List, Tuple, Optional, Generator


ALGOS = {
    "sac": SAC,
    "bc": bc,
}

RESULTS_PATH = Path.cwd() / "results"
BASE_PATH = RESULTS_PATH / "test"

PHANTOMS = ["phantom3"]
TARGETS = ["bca"]
CONFIGS = ["full"]


def create_experiment_directories(
    experiment_path: Path = None,
    base_path: Path = None,
) -> Tuple[Path, Path, Path]:
    assert experiment_path, "experiment_path must be specified"

    experiment_path = base_path or BASE_PATH / experiment_path
    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"

    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)

    return model_path, log_path, eval_path


def get_experiment_paths(path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    model_dir, log_dir, eval_dir = create_experiment_directories(path)

    models = sorted(model_dir.glob("*.zip"), key=lambda x: x.stem.split("_")[-1])
    logs = sorted(log_dir.rglob("*tfevents*"), key=lambda x: x.stem.split(".")[3])
    evals = sorted(eval_dir.rglob("*.npz"), key=lambda x: x.stem.split("_")[-1])

    return models, logs, evals


def get_tb_logs(
    path: Path,
    n_interpolations: int = 30,
    max_timesteps: int = 600_000,
    length_threshold: int = 150,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Inner function to parse tensorboard log.
    def parse_log(tb_path: Path) -> Optional[pd.DataFrame]:
        tag = "rollout/ep_len_mean"
        acc = EventAccumulator(tb_path.as_posix())
        acc.Reload()
        if tag not in acc.Tags()["scalars"]:
            print("Tag not found")
            return
        df = pd.DataFrame(acc.Scalars(tag))
        df = df.drop(columns=["wall_time"]).groupby("step").mean()
        return df

    _, log_paths, _ = get_experiment_paths(path)
    logs = []
    for log in log_paths:
        tensorboard_log = parse_log(log)
        if tensorboard_log is None or len(tensorboard_log) == 0:
            continue
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


def evaluate_all_models_in_path(
    experiments_path: Optional[Path] = None,
    n_episodes: int = 10,
    phantom_filter: Optional[str] = None,
    target_filter: Optional[str] = None,
    algorithm_filter: Optional[str] = None,
):
    if not experiments_path:
        experiments_path = BASE_PATH

    for algorithm_path in gather_paths(
        experiments_path, phantom_filter, target_filter, algorithm_filter
    ):
        evaluate_single_model(algorithm_path, n_episodes)


def gather_paths(
    root_path: Path,
    phantom_filter: Optional[str] = None,
    target_filter: Optional[str] = None,
    algorithm_filter: Optional[str] = None,
) -> Generator[Path, None, None]:
    """Generates algorithm paths based on a sequence of filters."""
    for phantom in filter_dirs(root_path, phantom_filter):
        for target in filter_dirs(phantom, target_filter):
            for algorithm in filter_dirs(target, algorithm_filter):
                yield algorithm


def filter_dirs(dir_path: Path, name_filter: Optional[str]) -> Union[Path]:
    return [
        child
        for child in dir_path.iterdir()
        if child.is_dir() and (not name_filter or child.name == name_filter)
    ]


def evaluate_single_model(algorithm_path: Path, n_episodes: int = 10):
    models, _, eval_path = get_experiment_paths(algorithm_path)

    for model_path in models:
        if not should_evaluate(model_path, eval_path):
            continue

        config = configure_algorithm(algorithm_path)
        env = make_gym_env(config)
        model_instance = load_model_instance(model_path, config, env)

        evaluation_data = evaluate_policy(model_instance, env, n_episodes)
        np.savez_compressed(eval_path / f"{model_path.stem}.npz", **evaluation_data)


def should_evaluate(model_path: Path, eval_path: Path) -> bool:
    return not (eval_path / f"{model_path.stem}.npz").exists()


def configure_algorithm(algorithm_path: Path) -> dict:
    config = get_config(algorithm_path.stem)
    config["task_kwargs"]["phantom"] = algorithm_path.parent.parent.stem
    config["task_kwargs"]["target"] = algorithm_path.parent.stem
    if "bc" in algorithm_path.stem:
        config["wrapper_kwargs"]["channel_first"] = True
    return config


def load_model_instance(model_path: Path, config: dict, env: gym.Env) -> BaseAlgorithm:
    if "bc" in model_path.stem:
        from scratch.bc.custom_networks import CnnPolicy
        from cathsim.wrappers import Dict2Array
        import torch as th

        env = Dict2Array(env)
        return CnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: th.finfo(th.float32).max,
        ).load(model_path)
    else:
        return SAC.load(model_path)


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
                models_path, logs_path, eva_path = get_experiment_paths(path)
                __import__("pprint").pprint(models_path)
                evaluate_all_models_in_path(path)
