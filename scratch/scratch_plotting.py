from cathsim.rl.utils import Config, generate_experiment_paths, EXPERIMENT_PATH
from evaluation.common import get_experiment_paths, parse_tb_log
from typing import Union, List, Callable, Dict
from cathsim.visualization import point2pixel

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from pprint import pprint
import matplotlib
import seaborn as sns

plt.style.use("test")
colors = sns.color_palette("deep")
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=colors)

PHANTOMS = ["phantom3", "low_tort", "phantom4"]
TARGETS = ["bca", "lcca"]
ALGORITHM_CONFIGS = ["full"]
BASE_IMAGES = {
    "phantom3": Path.cwd() / "figures" / "phantom3.png",
    # 'phantom4': Path.cwd().parent.parent / 'figures' / 'phantom4.png',
    # 'low_tort': Path.cwd().parent.parent / 'figures' / 'low_tort.png',
}

EXPERIMENT_NAMES_MAPPING = OrderedDict(
    {
        "pixels": "Image",
        "pixels_mask": "Image+Mask",
        "internal": "Internal",
        "internal_pixels": "Internal+Image",
        "full": "ENN",
        # 'full_w_her': 'AutoCath+HER',
        # 'full_w_her_w_sampling': 'AutoCath+HER+Sampling',
    }
)

PAGE_WIDTH = 5.50


class Trajectory:
    def __init__(self, keys=None, image_size=480, image=None):
        self.data = {key: [] for key in keys} if keys is not None else {}
        self.image_size = image_size
        self.image = (
            np.zeros((self.image_size, self.image_size, 3)) if image is None else image
        )

    def add_transition(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                raise ValueError(f"Invalid key: {key}")

    @staticmethod
    def from_dict(data):
        obj = Trajectory()
        obj.data = data
        return obj

    def plot_3D_to_2D(
        self, ax, key="head_positions", add_line: bool = False, **kwargs
    ) -> plt.Axes:
        data = self.data[key]
        camera_matrix = {
            480: np.array(
                [
                    [-5.79411255e02, 0.00000000e00, 2.39500000e02, -5.33073376e01],
                    [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.08351407e02],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, -1.50000000e-01],
                ]
            ),
            80: np.array(
                [
                    [-96.56854249, 0.0, 39.5, -8.82205627],
                    [0.0, 96.56854249, 39.5, -17.99606781],
                    [0.0, 0.0, 1.0, -0.15],
                ]
            ),
        }

        data = np.apply_along_axis(
            point2pixel, 1, data, camera_matrix=camera_matrix[80]
        )
        data = np.apply_along_axis(lambda x: x / 80 * 480, 1, data)
        data[:, 1] = 480 - data[:, 1]
        ax.scatter(data[:, 0], data[:, 1], **kwargs)
        return ax

    def flatten(self):
        new_dict = {}
        for key, value in self.data.items():
            if isinstance(value, dict):
                new_dict.update(self.flatten_dict(value))
            else:
                new_dict[key] = value
        return new_dict


class Trajectories:
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories

    def save(self, path: Path):
        np.savez(path, self.trajectories)


# checking if the base images exist
for base_image in BASE_IMAGES.values():
    if not base_image.exists():
        print(f"{base_image} does not exist")


def get_data(paths=List[Path], f: Callable = lambda x: x) -> List[np.ndarray]:
    return list(map(f, paths))


def process_eval_data(paths=List[Path]) -> Trajectories:
    def unpack(x):
        return [Trajectory.from_dict(y.item()) for y in x]

    def f(x):
        return dict(np.load(x, allow_pickle=True)).values()

    data = get_data(paths, f)
    trajectories = list(map(unpack, data))
    trajectories = [item for sublist in trajectories for item in sublist]
    return Trajectories(trajectories)


def get_human_data(path: Path, trial: int = 0) -> List[Path]:
    path = Path.cwd() / "human" / path / f"trial_{trial}"
    episodes = [episode for episode in path.iterdir()]
    for episode in episodes:
        trajectory = np.loadz(episode / "trajectory.npz")
        print(episode)


def process_human_data(
    paths=List[Path], mapping: Callable = lambda x: x
) -> Trajectories:
    def process_paths(paths):
        pass

    def f(x):
        return dict(np.load(x, allow_pickle=True)).values()

    data = get_data(paths, f)
    trajectories = list(map(process_paths, data))
    pass


def get_tb_log_data(paths=List[Path]) -> List[np.ndarray]:
    return list(map(parse_tb_log, paths))


#
# models_path, log_paths, eval_paths = get_paths(path)
# eval_path_0 = eval_paths[0]
# trajectories = process_eval_data(eval_paths)
# print(trajectories.trajectories[0].data.keys())
# trajectory_1 = trajectories.trajectories[2]
# trajectory_1.flatten()
# fig, ax = plt.subplots()
# base_image = plt.imread(BASE_IMAGES['phantom3'])
# # rotate the image
# base_image = np.rot90(base_image, k=3)
# base_image = np.rot90(base_image, k=3)
# base_image = np.flip(base_image, axis=1)
# ax.imshow(base_image)
# ax.set_xlim(0, 480)
# ax.set_ylim(0, 480)
# trajectory_1.plot_head_positions(ax=ax)
# plt.show()


if __name__ == "__main__":
    path = Path(PHANTOMS[0], TARGETS[0], ALGORITHM_CONFIGS[0])
