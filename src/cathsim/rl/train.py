import os
import torch as th
from pathlib import Path


from stable_baselines3 import PPO, SAC
from cathsim.rl.data import RESULTS_PATH

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}

RESULTS_PATH = Path.cwd() / "results"


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
    from cathsim.rl.evaluation import evaluate_policy, save_trajectories
    from cathsim.rl import Config, make_gym_env

    config = Config(
        config_name=config_name,
        target=target,
        phantom=phantom,
        trial_name=trial_name,
        base_path=base_path,
    )

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
            trajectories = evaluate_policy(model, env, n_episodes=2)
            save_trajectories(trajectories, eval_path / f"{algo}_{seed}")
        th.cuda.empty_cache()
