from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from tqdm import tqdm
from typing import OrderedDict
from pprint import pprint

from rl.utils import generate_experiment_paths
from rl.data import Trajectory
from rl.metrics import AGGREGATE_METRICS, INDIVIDUAL_METRICS

from stable_baselines3.common.base_class import BaseAlgorithm

from typing import List, Tuple

RESULTS_SUMMARY = Path.cwd() / "results-summary-test"


def get_paths(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    paths = list(path.rglob("*"))

    return paths


def collate_evaluation_data(path: Path) -> dict:
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
    # Apply individual metrics to each trajectory
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


def aggregate_results(
    eval_path: Path = None, output_path: Path = None, verbose: bool = False
) -> pd.DataFrame:
    eval_path = eval_path or RESULTS_SUMMARY

    if verbose:
        print(
            f'Analyzing {"experiment".ljust(30)} {"phantom".ljust(30)} {"target".ljust(30)}'
        )

    dataframe = pd.DataFrame()

    phantoms = [p for p in eval_path.iterdir() if p.is_dir()]
    for phantom in phantoms:
        targets = [t for t in phantom.iterdir() if t.is_dir()]
        for target in targets:
            files = [f for f in target.iterdir() if f.suffix == ".npz"]
            for file in files:
                print(
                    f"Analyzing {file.stem.ljust(30)} {phantom.stem.ljust(30)} {target.stem.ljust(30)}"
                )
                results = analyze_model(file)
                if results is not None:
                    if dataframe.empty:
                        dataframe = pd.DataFrame(
                            columns=["phantom", "target", "algorithm", *results.keys()]
                        )
                    results_dataframe = pd.DataFrame(
                        {
                            "phantom": phantom.stem,
                            "target": target.stem,
                            "algorithm": file.stem,
                            **results,
                        },
                        index=[0],
                    )
                    dataframe = pd.concat(
                        [dataframe, results_dataframe], ignore_index=True
                    )
    return dataframe


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
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation)
            new_observation, reward, done, info = env.step(action)
            trajectory.add_transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=new_observation,
                info=info,
            )
            observation = new_observation
        trajectories.append(trajectory)
    if n_episodes == 1:
        return trajectories[0]
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
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd

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


if __name__ == "__main__":
    # evaluate_model(EXPERIMENT_PATH / 'low_tort' / 'bca' / 'full', n_episodes=10)
    # evaluate_models()
    # lcca_evaluation.mkdir(exist_ok=True)

    # collate_results(verbose=True)
    eval_data = collate_evaluation_data(Path.cwd() / "test-base" / "test-trial")
    analysis = analyze_evaluation_data(eval_data)
    pprint(analysis)

    exit()
    dataframe = aggregate_results()
    exit()
    print(dataframe)
    dataframe.to_csv(RESULTS_SUMMARY / "results_4.csv", index=False)
    # make column names title case, without underscores
    dataframe.columns = [
        column.replace("_", " ").title() for column in dataframe.columns
    ]

    columns = [
        "Phantom",
        "Target",
        "Algorithm",
        "Force",
        "Force Std",
        "Path Length",
        "Path Length Std",
        "Episode Length",
        "Episode Length Std",
        "Safety",
        "Safety Std",
        "Curv",
        "Curv Std",
        "Success",
        "Success Std",
        "Spl",
    ]
    # remove curv and curv std
    columns.pop(11)
    columns.pop(12)
    # make sure all the numbers are formatted with two 00s after the decimal point
    # using :.2f
    dataframe = dataframe.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

    new_columns = [
        "Phantom",
        "Target",
        "Algorithm",
        "Force (N)",
        "Path Length (mm)",
        "Episode Length (s)",
        "Safety %",
        "Success %",
        "SPL %",
    ]
    dataframe["Force (N)"] = (
        "$"
        + dataframe["Force"].astype(str)
        + " \\pm "
        + dataframe["Force Std"].astype(str)
        + "$"
    )
    dataframe["Path Length (mm)"] = (
        "$"
        + dataframe["Path Length"].astype(str)
        + " \\pm "
        + dataframe["Path Length Std"].astype(str)
        + "$"
    )
    dataframe["Episode Length (s)"] = (
        "$"
        + dataframe["Episode Length"].astype(str)
        + " \\pm "
        + dataframe["Episode Length Std"].astype(str)
        + "$"
    )
    dataframe["Safety %"] = (
        "$"
        + dataframe["Safety"].astype(str)
        + " \\pm "
        + dataframe["Safety Std"].astype(str)
        + "$"
    )
    dataframe["Success %"] = (
        "$"
        + dataframe["Success"].astype(str)
        + " \\pm "
        + dataframe["Success Std"].astype(str)
        + "$"
    )
    dataframe["SPL %"] = "$" + dataframe["Spl"].astype(str) + "$"
    # format the elements of the columns
    # drop the row where the phantom is low_tort
    dataframe = dataframe[dataframe["Phantom"] != "low_tort"]
    dataframe = dataframe[dataframe["Phantom"] != "phantom4"]
    formatters = {
        "Phantom": lambda x: "Type-I Aortic Arch"
        if x == "phantom3"
        else "Type-II Aortic Arch",
        "Target": lambda x: x.upper(),
        "Algorithm": lambda x: x.replace("_", " ").title(),
    }
    # make the targets, which are bca and lcca, to be as a second collumn

    for column in new_columns:
        if column in formatters:
            dataframe[column] = dataframe[column].apply(formatters[column])

    # multiindex based on the phantom and target
    dataframe = dataframe.set_index(["Phantom", "Target", "Algorithm"])

    print(dataframe)
    print(
        dataframe.to_latex(
            float_format="%.2f",
            sparsify=True,
            columns=new_columns,
            column_format="cccrrrrrr",
            escape=False,
            formatters={
                "Phantom": lambda x: x.upper(),
                "Target": lambda x: x.upper(),
            },
        )
    )
