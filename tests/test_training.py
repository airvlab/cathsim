from cathsim.rl import train
from pathlib import Path

if __name__ == "__main__":
    algo = "sac"
    config_name = "test_full"
    target = "bca"
    phantom = "phantom3"
    trial_name = "test-trial"
    base_path = Path.cwd() / "my-test-results"
    n_timesteps = int(1e3)
    n_runs = 2
    evaluate = True

    train(
        algo=algo,
        config_name=config_name,
        target=target,
        phantom=phantom,
        trial_name=trial_name,
        base_path=Path.cwd() / base_path,
        n_timesteps=n_timesteps,
        n_runs=n_runs,
        evaluate=evaluate,
    )
