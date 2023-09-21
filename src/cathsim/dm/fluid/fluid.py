import numpy as np
import pandas as pd
from pathlib import Path
from dm_control import mujoco

data_path  = Path(__file__).parent / 'export1.csv'
DATA = pd.read_csv(data_path)

def get_guidewire_geom_info(physics) -> list[np.ndarray]:
    model = physics.copy().model
    guidewire_geom_ids = []
    for i in range(model.nbody):
        geom_name = model.body(i).name
        contains_guidewire = "guidewire" in geom_name
        is_part = "body" in geom_name
        if contains_guidewire and is_part:
            guidewire_geom_ids.append(model.geom(i).id)
    guidewire_geom_info = []
    for i in guidewire_geom_ids:
        position = physics.data.geom_xpos[i]
        guidewire_geom_info.append((i, position))
    return guidewire_geom_info

def find_nearest_speed(full_positions,data = DATA):
    forces = np.zeros((len(full_positions), 3))
    for idx, pos in enumerate(full_positions):
        distances = np.sqrt((data['X'] - pos[0]) ** 2 + 
                            (data['Y'] - pos[1]) ** 2 + 
                            (data['Z'] - pos[2]) ** 2)
        nearest_indices = distances.nsmallest(3).index
        avg_force = data.loc[nearest_indices, ['VX', 'VY', 'VZ']].mean().values
        forces[idx] = avg_force
    return forces

def apply_fluid_force(physics):
    guidewire_geom_info = get_guidewire_geom_info(physics)
    positions = [info[1] for info in guidewire_geom_info]
    joint_xpos = np.array(positions)
    forces = find_nearest_speed(joint_xpos)
    for idx, (i, pos) in enumerate(guidewire_geom_info):
        f = forces[idx, :].reshape(3, 1)
        torque = np.zeros_like(f)
        mujoco.mj_applyFT(
            physics.model.ptr,
            physics.data.ptr,
            f,
            torque,
            pos,
            i,
            physics.data.qfrc_applied,
        )