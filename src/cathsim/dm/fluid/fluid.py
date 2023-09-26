import numpy as np
from pathlib import Path
from dm_control import mujoco
from cathsim.dm.physics_functions import (
    get_geom_pos,
    get_bodies_pos,
)
from functools import lru_cache
from scipy.spatial import KDTree
from typing import Union

data_path = Path(__file__).parent / "data.csv"


@lru_cache(maxsize=None)
def get_data():
    return np.genfromtxt(data_path, delimiter=",", skip_header=1)


@lru_cache(maxsize=None)
def get_pos_and_vel():
    data = get_data()
    pos = data[:, :3]
    vel = data[:, 3:]
    assert pos.shape == vel.shape
    return pos, vel


@lru_cache(maxsize=None)
def get_tree():
    pos, _ = get_pos_and_vel()
    return KDTree(pos)


def find_average_velocity(
    query_position: Union[np.ndarray, list[np.ndarray]],
    n: int = 3,
    distance_threshold: float = 0.001,
) -> Union[np.ndarray, list[np.ndarray]]:
    tree = get_tree()
    _, vel = get_pos_and_vel()

    if isinstance(query_position, np.ndarray):
        distances, indices = tree.query(query_position, k=n)
        average_velocity = np.mean(vel[indices], axis=0)

        if distances[0] > distance_threshold:
            vel = np.zeros_like(average_velocity)
    elif isinstance(query_position, list):
        average_velocity = []
        for pos in query_position:
            average_velocity.append(find_average_velocity(pos))
    return average_velocity


def apply_fluid_force(physics, bodies_ids: list[int]):
    pos = get_bodies_pos(physics, bodies_ids)
    vel = find_average_velocity(pos, 3)
    torque = np.zeros_like(vel[0])
    for id, pos, vel in zip(bodies_ids, pos, vel):
        mujoco.mj_applyFT(
            physics.model.ptr,
            physics.data.ptr,
            vel,
            torque,
            pos,
            id,
            physics.data.qfrc_passive,
        )
