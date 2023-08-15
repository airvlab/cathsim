from pathlib import Path
from typing import Union
import pickle
import numpy as np
import gym
from stable_baselines3.common.base_class import BaseAlgorithm
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import pprint
from toolz.dicttoolz import itemmap
from functools import reduce
from cathsim.utils import point2pixel
from utils import flatten_dict, expand_dict, map_val

import warnings
from tqdm import tqdm


def plot_3D_to_2D(
    ax,
    data,
    base_image: np.ndarray = None,
    add_line: bool = True,
    image_size: int = 80,
    line_kwargs: dict = dict(color="black"),
    scatter_kwargs: dict = dict(color="blue"),
) -> plt.Axes:
    if base_image is not None:
        base_image = np.flipud(base_image)
        plt.imshow(base_image)
        image_size = base_image.shape[0]
        warnings.warn("Image size overwritten by base_image")

    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    if len(data[0]) == 3:
        data = np.apply_along_axis(
            point2pixel, 1, data, camera_kwargs=dict(image_size=image_size)
        )
        data = [point for point in data if np.all((0 <= point) & (point <= image_size))]
        data = np.array(data)  # Convert back to numpy array
        data[:, 1] = image_size - data[:, 1]
    ax.scatter(data[:, 0], data[:, 1], **scatter_kwargs)
    if add_line:
        ax.plot(data[:, 0], data[:, 1], **line_kwargs)


class Trajectory:
    def __init__(
        self,
        image_size: int = None,
        camera_matrix: np.ndarray = None,
        **kwargs,
    ):
        self.data = kwargs or self._initialize(kwargs)
        self.image_size = image_size
        self.camera_matrix = camera_matrix

    def __str__(self):
        def fn(item):
            k, v = item
            if isinstance(v, dict):
                return (k, itemmap(fn, v))
            else:
                return (k, v.shape if isinstance(v, np.ndarray) else np.array(v).shape)

        return pprint.pformat(itemmap(fn, self.data))

    def __len__(self):
        def fn(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    fn(v)
                else:
                    return len(v)

        return fn(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return map_val(lambda x: x[index], self.data)
        elif isinstance(index, str):

            def fn(acc, item):
                k, v = item
                if isinstance(v, dict):
                    new_items = reduce(fn, v.items(), {})
                    acc.update(new_items)
                elif isinstance(k, str) and index in k:
                    acc[k] = v
                return acc

            return reduce(fn, self.data.items(), {})

        else:
            raise TypeError("Invalid Argument Type")

    def get_k_len(self, key: str = None):
        def fn(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    fn(v)
                else:
                    if k == key:
                        return len(v)

        return fn(self.data)

    def _initialize(self, d: dict):
        self.data = map_val(lambda x: [], d)

    def _validate(self):
        def fn(acc, item):
            k, v = item
            if isinstance(v, dict):
                return all(fn(acc, sub_item) for sub_item in v.items())
            else:
                return len(v) if acc is None else len(v) == acc

        valid = reduce(fn, self.data.items(), len(self))
        if not valid:
            print(
                f"""Trajectory has uneven lengths. 
        If a final obs is stored, please remove it or create a new obs using 
        Trajectory.make_next_obs().
                   """
            )
            exit()
        return self

    @staticmethod
    def from_dict(data):
        obj = Trajectory()
        obj.data = data
        return obj

    def to_array(self):
        self.data = map_val(
            lambda x: x if isinstance(x, np.ndarray) else np.array(x), self.data
        )
        return self

    def add_transition(self, **kwargs):
        if self.data is None:
            self._initialize(kwargs)
        self.data = expand_dict(self.data, kwargs)

    def flatten(self):
        self.data = flatten_dict(self.data)
        return self

    def apply(self, fn: callable, key: str = None):
        def gn(item):
            k, v = item
            if isinstance(v, dict):
                return (k, itemmap(gn, v))
            else:
                if key is None:
                    return (k, fn(v))
                else:
                    return (k, fn(v) if k == key else v)

        self.data = itemmap(gn, self.data)

        return self

    def save(self, file_path: Union[str, Path]):
        if file_path.parent.exists() == False:
            file_path.parent.mkdir()
        with open(file_path, "wb") as file:
            pickle.dump(self.data, file)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return Trajectory.from_dict(data)

    def plot_path(
        self,
        ax,
        key: str = None,
        plot_kwargs: dict = dict(
            base_image=None,
            add_line=True,
            line_kwargs={},
            scatter_kwargs={},
        ),
    ):
        data = list(self[key or "head_pos"].values())
        plot_3D_to_2D(ax, *data, **plot_kwargs)


class TrajectoriesDataset(data.Dataset):
    def __init__(self, path: Path, transform_image=None, lazy_load=True):
        self.trajectories = list(path.iterdir())
        self.lazy_load = lazy_load
        if not lazy_load:
            self.trajectories = [
                Trajectory.load(p).to_array()["head_pos"] for p in self.trajectories
            ]

    def __len__(self):
        return len(self.trajectories)

    @staticmethod
    def patch_trajectory(traj: np.ndarray, length: int = 300) -> np.ndarray:
        shape = traj.shape
        shape = (length - len(traj), shape[1])
        return np.concatenate([traj, np.zeros(shape=shape)], axis=0)

    def __getitem__(self, idx):
        # TODO: fix the lazy loading
        if self.lazy_load:
            trajectory = Trajectory.load(self.trajectories[idx]).to_array()
        trajectory = self.trajectories[idx]["head_pos"]
        start = trajectory[0]
        goal = trajectory[-1]
        trajectory = self.patch_trajectory(trajectory)

        start = torch.from_numpy(start).float()
        goal = torch.from_numpy(goal).float()
        trajectory = torch.from_numpy(trajectory).float()

        return (start, goal), trajectory


def generate_trajectory(
    model: BaseAlgorithm, env: gym.Env, n_episodes: int = 10
) -> dict:
    trajectory = Trajectory()
    obs = env.reset()
    done = False
    while not done:
        act, _states = model.predict(obs)
        next_obs, reward, done, info = env.step(act)
        trajectory.add_transition(
            obs=obs, act=act, next_obs=next_obs, reward=reward, info=info
        )
        obs = next_obs
    return trajectory


def generate_trajectories(algorithm_path: Path, n_episodes: int = 10_000):
    from stable_baselines3 import SAC
    from rl.utils import get_config, make_experiment
    from cathsim.utils import make_gym_env

    model_path, _, eval_path = make_experiment(
        algorithm_path,
        base_path=Path.cwd() / Path("results/experiments/"),
    )
    print(model_path)

    for model_filename in model_path.iterdir():
        model_name = model_filename.stem
        print(f"Evaluating {model_name} in {algorithm_path} for {n_episodes} episodes.")
        config = get_config(algorithm_path.stem)
        config["task_kwargs"]["phantom"] = algorithm_path.parent.parent.stem
        config["task_kwargs"]["target"] = algorithm_path.parent.stem
        algo_kwargs = config["algo_kwargs"]
        env = make_gym_env(config)
        model = SAC.load(
            model_filename,
            custom_objects={"policy_kwargs": algo_kwargs.get("policy_kwargs", {})},
        )

        for n in tqdm(range(n_episodes)):
            trajectory = generate_trajectory(model, env)
            trajectory.save(Path(f"transitions_bodies-full/{n}"))


if __name__ == "__main__":
    generate_trajectories(Path("phantom3/bca/full"))
    exit()
    path = Path.cwd() / Path("transitions-shape/")
    trajectories = list(path.iterdir())
    traj_1 = Trajectory.load(trajectories[0]).to_array()
    print(traj_1)
    # print(len(traj_1))
    # print(traj_1.flatten())
    # print(traj_1)
    # traj_1 = traj_1.apply(lambda x: np.array(x))
    # print(traj_1.data["info"][0])
    # print(traj_1)
    td = TrajectoriesDataset(path, lazy_load=False)
    td_loader = data.DataLoader(td, batch_size=2)
    (start, goal), path = next(iter(td_loader))
