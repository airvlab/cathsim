from pathlib import Path
import pickle
import numpy as np
import gym
from stable_baselines3.common.base_class import BaseAlgorithm

import torch
from torch.utils import data
import pprint
from toolz import dicttoolz


def flatten_dict(d: dict) -> dict:
    def flatten_dict(d: dict, acc, parent_key: str = None, sep="-"):
        for k, v in d.items():
            if parent_key:
                k = parent_key + sep + k
            if isinstance(v, dict):
                flatten_dict(v, acc, k, sep)
            else:
                acc[k] = v
        return acc

    return flatten_dict(d, {})


class Trajectory:
    def __init__(self, keys=None, image_size=480):
        self.data = {key: [] for key in keys} if keys is not None else None

    def __str__(self):
        d = self.data.copy()
        d = dicttoolz.valmap(
            lambda x: x if isinstance(x, np.ndarray) else np.array(x), d
        )
        d = dicttoolz.valmap(lambda x: x.shape, d)
        return pprint.pformat(d)

    def __len__(self):
        dict_keys = list(self.data)
        return len(self.data[dict_keys[0]])

    def _initialize(self, keys):
        data = {}
        for key in keys:
            data[key] = []
        self.data = data

    @staticmethod
    def from_dict(data):
        obj = Trajectory()
        obj.data = data
        return obj

    def add_transition(self, **kwargs):
        if self.data is None:
            self._initialize(kwargs.keys())
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                raise ValueError(f"Invalid key: {key}")

    def flatten(self):
        self.data = flatten_dict(self.data)
        return self

    def apply(self, fn: callable, key: int = None):
        self.data = dicttoolz.itemmap(
            lambda item: (item[0], fn(item[1])) if item[0] == key else item, self.data
        )
        return self

    def save(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.data, file)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return Trajectory.from_dict(data)


class TrajectoriesDataset(data.Dataset):
    def __init__(self, trajectories, transform_image=None, lazy_load=True):
        self.trajectories = trajectories
        if not lazy_load:
            self.trajectories = [Trajectory.load(p) for p in self.trajectories]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        start = trajectory["obs"][0]
        goal = trajectory["info-goal"]
        path = trajectory["info-head_pos"]

        start = torch.from_numpy(start).float()
        goal = torch.from_numpy(goal).float()
        path = torch.from_numpy(path).float()

        return (start, goal), path


def generate_trajectory(
    model: BaseAlgorithm, env: gym.Env, n_episodes: int = 10
) -> dict:
    trajectory = Trajectory()
    obs = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs)
        next_obs, reward, done, info = env.step(act)
        trajectory.add_transition(
            obs=obs, act=act, next_obs=next_obs, reward=reward, info=info
        )
    return trajectory


def generate_trajectories(algorithm_path: Path, n_episodes: int = 2):
    from stable_baselines3 import SAC
    from rl.utils import get_config, make_experiment
    from cathsim.cathsim.env_utils import make_gym_env

    model_path, _, eval_path = make_experiment(algorithm_path)

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
        for n in range(n_episodes):
            trajectory = generate_trajectory(model, env)
            trajectory.save(Path(f"./scratch/transitions/{n}"))
        exit()


if __name__ == "__main__":
    trajectories_path = Path.cwd() / Path("scratch/transitions/")
    trajectories = list(trajectories_path.iterdir())
    td = TrajectoriesDataset(trajectories=trajectories, lazy_load=False)
    print(len(td))
    # generate_trajectories(Path("phantom3/bca/full/"), 20)
    # t = Trajectory().load(Path("./scratch/transitions/0"))
    # t = t.apply(lambda x: np.array(x)).apply(lambda x: x.shape)
    # print(t)
