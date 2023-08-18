import os
import torch as th
import yaml
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from mergedeep import merge

from stable_baselines3 import PPO, SAC

from imitation.algorithms import bc
from cathsim.utils import make_gym_env

import pprint

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "bc": bc,
}

RESULTS_PATH = Path.cwd() / "results"


class Config:
    def __init__(
        self,
        config_name: str = None,
        target: str = "bca",
        phantom: str = "phantom3",
        trial_name: str = "test",
        base_path: Path = RESULTS_PATH,
        task_kwargs: dict = {},
        wrapper_kwargs: dict = {},
        algo_kwargs: dict = {},
    ):
        from rl.custom_extractor import CustomExtractor

        self.base_path = base_path
        self.trial_name = trial_name

        self.config_name = config_name
        self.task_kwargs = task_kwargs

        self.task_kwargs = dict(
            image_size=80,
            phantom=phantom,
            target=target,
        )
        merge(self.task_kwargs, task_kwargs)

        self.wrapper_kwargs = dict(
            time_limit=300,
            grayscale=True,
            channels_first=True,
        )
        merge(self.wrapper_kwargs, wrapper_kwargs)

        self.algo_kwargs = dict(
            buffer_size=int(5e5),
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=CustomExtractor,
            ),
            device="cuda" if th.cuda.is_available() else "cpu",
        )
        merge(self.algo_kwargs, algo_kwargs)

        if config_name:
            self.load(config_name)

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __add__(self, other):
        new_config = Config(None)
        new_config.__dict__ = merge(self.__dict__, other.__dict__)
        return new_config

    def load(self, config_name: str):
        with open(Path(__file__).parent / "config" / (config_name + ".yaml"), "r") as f:
            config = yaml.safe_load(f)
        merge(self.__dict__, config)

    def update(self, config: dict):
        merge(self.__dict__, config)

    def get_env_path(self):
        return (
            self.base_path
            / self.trial_name
            / Path(
                f"{self.task_kwargs['phantom']}/{self.task_kwargs['target']}/{self.config_name}"
            )
        )


def make_path(
    config_name: str,
    phantom: str,
    target: str,
    trial_name: str = None,
    base_path: Path = None,
):
    pass


def generate_experiment_paths(experiment_path: Path = None) -> tuple:
    if experiment_path.is_absolute() is False:
        experiment_path = RESULTS_PATH / experiment_path

    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for directory_path in [experiment_path, model_path, log_path, eval_path]:
        directory_path.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def train(
    algo: str,
    config_name: str = "test",
    target: str = "bca",
    phantom: str = "phantom3",
    trial_name: str = "test",
    base_path: Path = RESULTS_PATH,
    n_timesteps: int = 600_000,
    n_runs: int = 4,
    evaluate: bool = False,
    n_envs: int = None,
    **kwargs,
) -> None:
    """Train a model.

    This function trains a model using the specified algorithm and configuration.

    Args:
        algo (str): Algorithm to use. Currently supported: ppo, sac
        config_name (str): The name of the configuration file (see config folder)
        target (str): The target to use. Currently supported: bca, lcca
        phantom (str): The phantom to use.
        trial_name (str): The trial name to use. Used to separate different runs.
        base_path (Path): The base path to use for saving the results.
        n_timesteps (int): Number of timesteps to train for.
        n_runs (int): Number of runs to train.
        evaluate (bool): Flag to evaluate the model after training.
        n_envs (int): Number of environments to use for training. Defaults to half the number of CPU cores.
    """
    from rl.evaluation import evaluate_policy, save_trajectories

    config = Config(config_name, target, phantom, trial_name, base_path)

    experiment_path = config.get_env_path()
    model_path, log_path, eval_path = generate_experiment_paths(experiment_path)

    env = make_gym_env(config=config, n_envs=n_envs or os.cpu_count() // 2)

    for seed in range(n_runs):
        if (model_path / f"{algo}_{seed}.zip").exists():
            print(f"Model {algo}_{seed} already exists, skipping")
            continue

        model = ALGOS[algo](
            env=env,
            verbose=1,
            tensorboard_log=log_path,
            **config.algo_kwargs,
        )

        model.learn(
            total_timesteps=n_timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"{algo}_{seed}",
            log_interval=10,
        )

        model.save(model_path / f"{algo}_{seed}.zip")

        if evaluate:
            env = make_gym_env(config=config, monitor_wrapper=False)
            trajectories = evaluate_policy(model, env, n_episodes=10)
            save_trajectories(trajectories, eval_path / f"{algo}_{seed}")
        th.cuda.empty_cache()


def load_sb3_model(path: Path, config_name: str = None) -> BaseAlgorithm:
    """Load the model with custom policy objects if needed.

    Args:
        model_filename (Path): Path to the model file.
        config_name (str): Name of the algorithm for configuration retrieval.

    Returns:
        BaseAlgorithm: Loaded model.
    """
    config = Config(config_name)
    algo_kwargs = config.get("algo_kwargs", {})

    # Load the model with custom policy if required
    model = SAC.load(
        path,
        custom_objects={"policy_kwargs": algo_kwargs.get("policy_kwargs", {})},
    )
    return model


if __name__ == "__main__":
    config = Config("pixels")
    print(config)
