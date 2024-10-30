import numpy as np
from cathsim.rl.data import Trajectory
from typing import List

from cathsim.dm.utils import distance


def calculate_total_distance(positions):
    return np.sum(distance(positions[1:], positions[:-1]))


def force_mean(trajectory: Trajectory) -> float:
    return trajectory["info-forces"].mean()


def reward_mean(trajectory: Trajectory) -> float:
    return trajectory["reward"].mean()


def force_max(trajectory: Trajectory) -> float:
    return trajectory["info-forces"].max()


def head_pos_mean(trajectory: Trajectory) -> float:
    return np.mean(trajectory["info-head_pos"], axis=0)


def episode_length(trajectory: Trajectory) -> float:
    return len(trajectory["info-head_pos"])


def safety(trajectory: Trajectory) -> float:
    return 1 - np.sum(np.where(trajectory["info-forces"] > 2, 1, 0)) / episode_length(
        trajectory
    )


def total_distance(trajectory: Trajectory) -> float:
    return calculate_total_distance(trajectory["info-head_pos"]) * 100


def success(trajectory: Trajectory) -> float:
    return np.sum(np.where(episode_length(trajectory) < 300, 1, 0))


def spl(trajectories: List[Trajectory]) -> float:
    aggregated_distance = sum(
        [total_distance(traj) for traj in trajectories]
    )  # renamed variable
    total_success = sum([success(traj) for traj in trajectories])
    num_trajectories = len(trajectories)
    optimal_path_length = 15.73  # This could also be parameterized

    return (total_success / num_trajectories) * (
        optimal_path_length
        / max(aggregated_distance / num_trajectories, optimal_path_length)
    )


INDIVIDUAL_METRICS = [
    force_mean,
    force_max,
    reward_mean,
    # head_pos_mean,
    episode_length,
    safety,
    total_distance,
    success,
]
AGGREGATE_METRICS = [spl]
