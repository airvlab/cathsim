from pathlib import Path
import pickle
import numpy as np
import gym
from stable_baselines3.common.base_class import BaseAlgorithm

import torch
from torch.utils import data
import pprint
from toolz import dicttoolz
from toolz.dicttoolz import itemmap
from cathsim.utils.common import flatten_dict, expand_dict


class Trajectory:
    def __init__(self, image_size=480, **kwargs):
        self.data = kwargs or self._initialize(kwargs)

    def __str__(self):
        def fn(item):
            k, v = item
            if isinstance(v, dict):
                return (k, itemmap(fn, v))
            else:
                return (k, v.shape if isinstance(v, np.ndarray) else np.array(v).shape)

        return pprint.pformat(itemmap(fn, self.data))

    def __len__(self):
        def find_len(d: dict):
            for k, v in self.data.items():
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    return len(v)
                elif isinstance(v, dict):
                    return find_len(v)
                else:
                    raise TypeError(f"dict_val is not a np.ndarray or list: {type(v)}")

        return find_len(self.data)

    def _initialize(self, d: dict):
        def fn(item):
            k, v = item
            if isinstance(v, dict):
                return (k, itemmap(fn, v))
            else:
                return (k, [])

        self.data = itemmap(fn, d)

    @staticmethod
    def from_dict(data):
        obj = Trajectory()
        obj.data = data
        return obj

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
        trajectory.add_transition(obs=obs, act=act, reward=reward, info=info)
    return trajectory


def generate_trajectories(algorithm_path: Path, n_episodes: int = 20):
    from stable_baselines3 import SAC
    from rl.utils import get_config, make_experiment
    from cathsim.cathsim.env_utils import make_gym_env

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
        for n in range(n_episodes):
            trajectory = generate_trajectory(model, env)
            print(trajectory)
            trajectory.save(Path(f"transitions/{n}"))
        exit()


if __name__ == "__main__":
    # generate_trajectories(Path("phantom3/bca/full"))
    trajectories_path = Path.cwd() / Path("transitions/")
    trajectories = list(trajectories_path.iterdir())
    traj_1 = Trajectory.load(trajectories[0]).apply(lambda x: print(len(x)))
    # print(traj_1.flatten())
    # print(traj_1)
    # traj_1 = traj_1.apply(lambda x: np.array(x))
    # print(traj_1.data["info"][0])
    # print(traj_1)
    # td = TrajectoriesDataset(trajectories=trajectories, lazy_load=False)
    # td_loader = data.DataLoader(td, batch_size=2)
