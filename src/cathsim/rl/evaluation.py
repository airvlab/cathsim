import tracemalloc
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cathsim.rl.data import Trajectory
from cathsim.rl.metrics import AGGREGATE_METRICS, INDIVIDUAL_METRICS
from cathsim.rl.utils import generate_experiment_paths
from memory_profiler import profile
from pympler.classtracker import ClassTracker
from pympler.tracker import SummaryTracker
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

RESULTS_SUMMARY = Path.cwd() / "results-summary-test"


def get_paths(path: Path) -> List[Path]:
    """Get all paths in a directory.

    This function recursively gets all paths in a directory.

    Args:
        path (Path): Path to the directory.

    Returns:
        List[Path]: List of paths in the directory.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    paths = list(path.rglob("*"))

    return paths


def collate_evaluation_data(path: Path) -> dict:
    """Helper function to collate evaluation data.

    This function collates evaluation data from a directory. The directory
    structure should be as follows:
        ``<phantom>/<target>/<config>/<algorithm>_<seed>/<trajectory>.pkl``

    Args:
        path (Path): The path to the directory containing the evaluation data. (this is the trial directory)

    Returns:
        dict: A dictionary containing the paths to the evaluation data.
    """

    def f(path: Path) -> bool:
        return path.suffix == ".pkl"

    paths = list(filter(f, get_paths(path)))

    evaluation_data = {}

    for path in paths:
        trajectory = Trajectory.load(path)
        phantom = path.parents[4].stem
        target = path.parents[3].stem
        config = path.parents[2].stem
        algorithm = path.parent.stem.split("_")[0]
        seed = path.parent.stem.split("_")[1]
        print(f"Collating {phantom}/{target}/{config}/{algorithm}/{seed}")
        evaluation_data.setdefault(config, {})
        evaluation_data[config].setdefault(phantom, {})
        evaluation_data[config][phantom].setdefault(target, {})
        evaluation_data[config][phantom][target].setdefault(seed, [])
        evaluation_data[config][phantom][target][seed].append(
            trajectory.flatten().to_array()
        )

    return evaluation_data


def analyze_and_aggregate(trajectories: List[Trajectory]) -> dict:
    """Helper function to analyze and aggregate trajectories.

    This function analyzes and aggregates a list of trajectories. It returns a
    dictionary containing the results of the analysis.

    Args:
        trajectories (List[Trajectory]): The trajectories to analyze and aggregate. The trajectories should be flattened. See Trajectory.flatten().

    Returns:
        dict: A dictionary containing the results of the analysis.
    """
    individual_metric_values = {metric.__name__: [] for metric in INDIVIDUAL_METRICS}

    for traj in trajectories:
        for metric in INDIVIDUAL_METRICS:
            individual_metric_values[metric.__name__].append(metric(traj))

    results = {}
    for metric, values in individual_metric_values.items():
        results[metric] = dict(mean=np.mean(values), std=np.std(values))

    for func in AGGREGATE_METRICS:
        results[func.__name__] = func(trajectories)

    return results


def analyze_evaluation_data(evaluation_data: dict) -> dict:
    """Analyze evaluation data.

    Analyzes evaluation data and returns a dictionary containing the results of
    the analysis.

    Args:
        evaluation_data (dict): The evaluation data to analyze. Results from collate_evaluation_data().

    Returns:
        dict: A dictionary containing the results of the analysis.
    """
    analysis_data = {}

    for config, config_data in evaluation_data.items():
        analysis_data[config] = {}
        for phantom, phantom_data in config_data.items():
            analysis_data[config][phantom] = {}
            for target, target_data in phantom_data.items():
                all_trajectories = [
                    trajectory.data
                    for seed, trajectories in target_data.items()
                    for trajectory in trajectories
                ]
                analysis_data[config][phantom][target] = analyze_and_aggregate(
                    all_trajectories
                )

    return analysis_data


@profile
def evaluate_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int = 1,
) -> Trajectory:
    """Evaluate a policy.

    This function evaluates a policy by running it in the environment until the
    episode is done.

    Args:
        model (BaseAlgorithm): A model that can predict actions given an observation.
        env (gym.Env): A gym environment.
        n_episodes (int): The number of episodes to evaluate the policy for.

    Returns:
        Trajectory: The trajectory of the episode.
    """
    trajectories = []

    for i in tqdm(range(n_episodes)):
        trajectory = Trajectory()
        observation, _ = env.reset()
        done = False
        j = 0
        while not done:
            action, _ = model.predict(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            continue
            print(
                f"Step {j} terminated: {terminated}, truncated: {truncated}, done: {done}",
                flush=True,
                end="\r",
            )
            trajectory.add_transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=new_observation,
                info=info,
            )
            observation = new_observation
            j += 1
    #     trajectories.append(trajectory)
    # if n_episodes == 1:
    #     return trajectories[0]

    exit()
    return trajectories


def save_trajectories(
    trajectories: List[Trajectory], path: Path, file_prefix: str = None
):
    """Save trajectories to a path.

    Args:
        trajectories (List[Trajectory]): A list of trajectories.
        path (Path): The path to save the trajectories to.
        file_prefix (str): A prefix to add to the filename.
    """
    path.mkdir(parents=True, exist_ok=True)
    for i, trajectory in enumerate(trajectories):
        filename = f"{file_prefix}_{i}.npz" if file_prefix else f"{i}.npz"
        if (path / filename).exists():
            raise FileExistsError(f"{path / filename} already exists.")
        trajectory.save(path / filename)


def parse_tensorboard_log(path: Path):
    """Parse a tensorboard log.

    :param path Path: The path to the tensorboard log.
    :return pd.DataFrame: The parsed tensorboard log.
    """
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tag = "rollout/ep_len_mean"
    acc = EventAccumulator(path)
    acc.Reload()
    # check if the tag exists
    if tag not in acc.Tags()["scalars"]:
        print("Tag not found")
        return
    df = pd.DataFrame(acc.Scalars(tag))
    df = df.drop(columns=["wall_time"]).groupby("step").mean()
    return df


def get_experiment_tensorboard_logs(
    path: Path, n_interpolations: int = None
) -> Tuple[List[pd.DataFrame], Tuple]:
    """Get the tensorboard logs for an experiment.

    :param path Path: The path to the experiment.
    :param n_interpolations int: The number of interpolations to use.
    :return List[pd.DataFrame]: A list of dataframes containing the tensorboard logs.
    """

    _, log_path, _ = generate_experiment_paths(path)
    logs = []
    log_paths = [log for log in log_path.iterdir() if log.is_dir()]
    for log in log_paths:
        tensorboard_log = parse_tensorboard_log(log.as_posix())
        assert len(tensorboard_log) != 0, f"Log {log} is empty."
        if len(tensorboard_log) < 150:
            print(
                f"Log {log.parent.parent.stem}-{log.stem} is too short({len(tensorboard_log)} < 500)."
            )
            continue
        logs.append(tensorboard_log)

    if n_interpolations:
        try:
            logs = [
                log.reindex(
                    np.arange(0, 600_000, 600_000 / n_interpolations), method="nearest"
                )
                for log in logs
            ]
            if len(logs) == 0:
                mean = None
                stdev = None
            else:
                mean = pd.concat(logs, axis=0).groupby(level=0).mean().squeeze()
                stdev = pd.concat(logs, axis=0).groupby(level=0).std().squeeze()
        except Exception as e:
            print(e)
            mean = None
            stdev = None
    return logs, (mean, stdev)


def plot_error_line_graph(
    ax: plt.Axes,
    mean: pd.Series,
    std: pd.Series,
    color: str = "C0",
    label: str = None,
    **kwargs,
):
    x = mean.index
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=color, **kwargs)


def plot_human_line(ax, phantom, target, n_interpolations=30):
    experiment_data = pd.read_csv(RESULTS_SUMMARY / "results_2.csv")
    human_data = experiment_data[
        (experiment_data["algorithm"] == "human")
        & (experiment_data["phantom"] == phantom)
        & (experiment_data["target"] == target)
    ]
    mean = human_data["episode_length"].to_numpy()[0]
    y_std = human_data["episode_length_std"].to_numpy()[0]

    x = np.arange(0, 600_000, 600_000 / n_interpolations)
    y = np.full(n_interpolations, mean)
    ax.plot(x, y + y_std, linewidth=0.5, color="C0")
    ax.plot(x, y - y_std, linewidth=0.5, color="C0")

    ax.plot(x, y, label="Human", linestyle="--", color="C0", linewidth=1)
    ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)


def evaluate_models():
    from cathsim.rl import make_gym_env
    from cathsim.rl.config_manager import Config
    from cathsim.rl.utils import generate_experiment_paths
    from stable_baselines3 import SAC

    config = Config(
        config_name="test",
        trial_name="test-trial",
        base_path=Path.cwd() / "results",
        task_kwargs=dict(
            phantom="phantom2",
            target="bca",
        ),
    )
    model_path, log_path, eval_path = generate_experiment_paths(
        config.get_env_path(),
    )
    models = config.get_env_path() / "models"
    for model_path in models.iterdir():
        model = SAC.load(model_path)
        if (eval_path / f"{model_path.stem}").exists():
            print(f"{model_path.stem} already exists.")
            continue
        trajectories = evaluate_policy(
            model, make_gym_env(config=config, monitor_wrapper=False), n_episodes=2
        )
        save_trajectories(
            trajectories,
            eval_path / f"{model_path.stem}",
        )
        del model


if __name__ == "__main__":
    # evaluate_models()
    # exit()

    # collate_results(verbose=True)
    eval_data = collate_evaluation_data(Path.cwd() / "results" / "test-trial")
    analysis = analyze_evaluation_data(eval_data)
    pprint(analysis)
