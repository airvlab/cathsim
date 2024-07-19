import os
import torch as th
from pathlib import Path


from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from cathsim.rl.data import RESULTS_PATH

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}


def evaluate_real(
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
    
    model = SAC.load(model_path / f"{algo}_{seed}.zip",
                     custom_objects={
                         "policy_kwargs": algo_kwargs.get("policy_kwargs", {})},
                     )
    
    trajectories = evaluate_policy(model, env, n_episodes=2)
    save_trajectories(trajectories, eval_path / f"{algo}_{seed}")
    th.cuda.empty_cache()
