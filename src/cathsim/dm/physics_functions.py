from dm_control.mujoco.engine import Physics
from dm_control import mujoco
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=None)
def get_guidewire_geom_ids(model: mujoco.MjModel) -> list[int]:
    model = model.copy()
    guidewire_geom_ids = []
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        is_guidewire_part = "guidewire" in geom_name
        if is_guidewire_part:
            guidewire_geom_ids.append(model.geom(i).id)

    return guidewire_geom_ids


def get_geom_pos(
    physics: Physics,
    geom_ids: list[int],
) -> list[np.ndarray]:
    data = physics.data.copy()
    positions = []
    for id in geom_ids:
        positions.append(data.geom_xpos[id])
    return positions


def get_guidewire_bodies_ids(model: mujoco.MjModel) -> list[int]:
    model = model.copy()
    guidewire_body_ids = []
    for i in range(model.nbody):
        body_name = model.body(i).name
        is_guidewire_part = "guidewire" in body_name
        is_body_part = "body" in body_name
        if is_guidewire_part and is_body_part:
            guidewire_body_ids.append(model.body(i).id)
    return guidewire_body_ids


def get_bodies_pos(
    physics: Physics,
    body_ids: list[int],
) -> list[np.ndarray]:
    data = physics.data.copy()
    positions = []
    for id in body_ids:
        positions.append(data.xpos[id])
    return positions


if __name__ == "__main__":
    from cathsim.dm import make_dm_env
    from pprint import pprint as pp

    env = make_dm_env("phantom3")
    physics = env.physics
    guidewire_geom_ids = get_guidewire_geom_ids(physics.model)
    assert len(guidewire_geom_ids) == 84
    guidewire_body_ids = get_guidewire_bodies_ids(physics.model)
    assert len(guidewire_body_ids) == 84
    guidewire_geom_pos = get_geom_pos(physics, guidewire_geom_ids)
    assert len(guidewire_geom_pos) == 84
    guidewire_body_pos = get_bodies_pos(physics, guidewire_body_ids)
    assert len(guidewire_body_pos) == 84
