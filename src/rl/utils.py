import os
import torch as th
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO, SAC, HerReplayBuffer

from imitation.algorithms import bc
from stable_baselines3.common import policies
from cathsim.wrappers import Dict2Array
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym

from cathsim.utils import make_gym_env


def flatten_dict(d: dict, parent_key: str = None) -> dict:
    acc = {}
    for k, v in d.items():
        if parent_key:
            k = parent_key + "-" + k
        if isinstance(v, dict):
            acc = acc | flatten_dict(v, k)
        else:
            acc[k] = v
    return acc


def expand_dict(xd: dict, yd: dict) -> dict:
    zd = xd.copy()
    for k, v in yd.items():
        if k not in xd:
            zd[k] = [v]
        elif isinstance(v, dict) and isinstance(xd[k], dict):
            zd[k] = expand_dict(xd[k], v)
        else:
            zd[k] = xd[k] + [v]
    return zd


def map_val(g: callable, d: dict):
    def f(item):
        k, v = item
        if isinstance(v, dict):
            return (k, itemmap(f, v))
        else:
            return (k, g(v))

    return itemmap(f, d)


class CnnPolicy(policies.ActorCriticCnnPolicy):
    """
    A CNN policy for behavioral clonning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "bc": bc,
}

RESULTS_PATH = Path(__file__).parent.parent / "results"
EXPERIMENT_PATH = RESULTS_PATH / "experiments"
EVALUATION_PATH = RESULTS_PATH / "evaluation"


def make_experiment(experiment_path: Path = None, base_path: Path = None) -> tuple:
    """Creates the maths for an experiment

    :param experiment_path: Path:  (Default value = None)
    :param base_path: Path:  (Default value = None)

    """
    assert experiment_path, "experiment_path must be specified"
    base_path = base_path or EXPERIMENT_PATH
    experiment_path = base_path / experiment_path
    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def make_env(
    n_envs: int = 1, config: dict = {}, monitor_wrapper: bool = True, **kwargs
):
    """Makes  gym environment given a configuration.

    :param n_envs: int:  (Default value = 1)
    :param config: dict:  (Default value = {})
    :param monitor_wrapper: bool:  (Default value = True)
    :param **kwargs:

    """

    def _init() -> gym.Env:
        env = make_gym_env(config=config)
        return env

    if n_envs == 1:
        env = DummyVecEnv([_init])
    else:
        env = SubprocVecEnv([_init for _ in range(n_envs)])
    if monitor_wrapper:
        if n_envs == 1:
            from stable_baselines3.common.monitor import Monitor

            env = Monitor(env)
        else:
            from stable_baselines3.common.vec_env import VecMonitor

            env = VecMonitor(env)

    return env


def train(
    algo: str,
    phantom: str = "phantom3",
    target: str = "bca",
    config_name: str = "test",
    base_path: Path = RESULTS_PATH,
    n_runs: int = 4,
    time_steps: int = 500_000,
    evaluate: bool = False,
    device: str = None,
    n_envs: int = None,
    vec_env: bool = True,
    config: dict = {},
    **kwargs,
) -> None:
    """Starts the training for an algorithm

    :param algo: str:
    :param experiment: str:
    :param experiment_path: Path:  (Default value = None)
    :param n_runs: int:  (Default value = 4)
    :param time_steps: int:  (Default value = 500_000)
    :param evaluate: bool:  (Default value = False)
    :param device: str:  (Default value = None)
    :param n_envs: int:  (Default value = None)
    :param vec_env: bool:  (Default value = True)
    :param config: dict:  (Default value = {})
    :param **kwargs:

    """
    from rl.evaluation import evaluate_policy

    algo_kwargs = config.get("algo_kwargs", {})

    if not device:
        device = "cuda" if th.cuda.is_available() else "cpu"
    n_envs = n_envs or os.cpu_count() // 2

    assert algo in ALGOS.keys(), f"algo must be one of {ALGOS.keys()}"
    assert n_runs > 0, "n_runs must be greater than 0"
    assert time_steps > 0, "time_steps must be greater than 0"
    assert device in ["cpu", "cuda"], "device must be one of [cpu, cuda]"
    assert n_envs > 0, "n_envs must be greater than 0"

    experiment_path = Path(f"{phantom}/{target}/{config_name}")
    model_path, log_path, eval_path = make_experiment(
        experiment_path, base_path=base_path
    )

    n_envs = n_envs or os.cpu_count() // 2

    env = make_env(n_envs=n_envs, config=config)

    for seed in range(n_runs):
        if (model_path / f"{algo}_{seed}.zip").exists():
            print(f"Model {algo} {seed} already exists, skipping")
            pass
        else:
            for key, value in algo_kwargs.items():
                __import__("pprint").pprint(f"{key}: {value}")
            model = ALGOS[algo](
                env=env,
                device=device,
                verbose=1,
                tensorboard_log=log_path,
                **algo_kwargs,
            )

            model.learn(
                total_timesteps=time_steps,
                log_interval=10,
                tb_log_name=f"{algo}_{seed}",
                progress_bar=True,
                reset_num_timesteps=False,
            )

            model.save(model_path / f"{algo}_{seed}.zip")

            if evaluate:
                results = evaluate_policy(model, env, 10)
                np.savez_compressed(eval_path / f"{algo}_{seed}", **results)
            th.cuda.empty_cache()


def cmd_visualize_agent(args=None):
    """Visualize a trained agent.

    :param args:  (Default value = None)

    """
    import cv2
    from cathsim.utils import make_gym_env
    from cathsim.utils import point2pixel
    import argparse as ap

    # from scratch.bc.custom_networks import CustomPolicy
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, default="full")
    parser.add_argument("--base-path", type=str, default="results")
    parser.add_argument("--trial", type=str, default="1")
    parser.add_argument("--phantom", type=str, default="phantom3")
    parser.add_argument("--target", type=str, default="bca")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-video", type=bool, default=False)
    parser.add_argument("--get_images", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--visualize-sites", type=bool, default=False)
    args = parser.parse_args()

    algo = args.algo

    if args.save_video:
        import moviepy.editor as mpy

    path = Path(f"{args.phantom}/{args.target}/{args.config}")
    config = get_config(args.config)
    model_path, log_path, eval_path = make_experiment(
        path, base_path=Path.cwd() / args.base_path / args.trial
    )

    if args.seed is None:
        model_path = model_path / algo
    else:
        model_path = model_path / (algo + f"_{args.seed}.zip")
    video_path = model_path.parent.parent / "videos"
    images_path = model_path.parent.parent / "images"

    assert model_path.exists(), f"{model_path} does not exist"

    config["task_kwargs"]["target"] = args.target
    config["task_kwargs"]["phantom"] = args.phantom
    config["task_kwargs"]["visualize_sites"] = args.visualize_sites

    if algo == "bc":
        config["wrapper_kwargs"]["channel_first"] = True
        env = make_gym_env(config=config)
        env = Dict2Array(env)
        model = CnnPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: th.finfo(th.float32).max,
        ).load(model_path)
    else:
        env = make_gym_env(config=config)
        model = ALGOS[algo].load(model_path)

    for episode in range(10):
        obs = env.reset()
        done = False
        frames = []
        segment_frames = []
        rewards = []
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)

            image = env.render("rgb_array", image_size=480)
            frames.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.circle(
                image,
                point2pixel(info["target_pos"], dict(image_size=480)),
                radius=10,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.imshow("image", image)
            cv2.waitKey(1)

        if args.save_video:
            os.makedirs(video_path, exist_ok=True)
            frames = [mpy.ImageClip(m).set_duration(0.1) for m in frames]
            clip = mpy.concatenate_videoclips(frames, method="compose")
            clip.write_videofile(
                (video_path / f"{algo}_{episode}.mp4").as_posix(), fps=60
            )
        if args.get_images:
            import matplotlib.pyplot as plt

            os.makedirs(images_path, exist_ok=True)
            print(f"Saving images to {images_path}")
            for i in range(len(frames)):
                plt.imsave(f"{images_path}/{algo}_{episode}_{i}.png", frames[i])
                plt.imsave(
                    f"{images_path}/{algo}_{episode}_{i}_segment.png",
                    segment_frames[i],
                    cmap="gray",
                )


def get_config(config_name: str) -> dict:
    """Parses a configuration file

    :param config_name: str:

    """
    import yaml
    from mergedeep import merge
    from rl.custom_extractor import CustomExtractor

    configs_path = Path(__file__).parent / "config"
    main_config_path = configs_path / "main.yaml"
    config_path = configs_path / (config_name + ".yaml")
    main_config = yaml.safe_load(open(main_config_path, "r"))
    config = yaml.safe_load(open(config_path, "r"))
    config = merge(main_config, config)

    policy_kwargs = main_config["algo_kwargs"].get("policy_kwargs", {})
    feature_extractor_class = policy_kwargs.get("features_extractor_class", None)
    if feature_extractor_class == "CustomExtractor":
        main_config["algo_kwargs"]["policy_kwargs"][
            "features_extractor_class"
        ] = CustomExtractor

    if main_config["algo_kwargs"].get("replay_buffer_class", None) == "HerReplayBuffer":
        main_config["algo_kwargs"]["replay_buffer_class"] = HerReplayBuffer
    return main_config


def process_human_trajectories(
    path: Path, flatten: bool = False, mapping: dict = None
) -> np.ndarray:
    """Utility function that processes human trajectories.

    :param path: Path:
    :param flatten: bool:  (Default value = False)
    :param mapping: dict:  (Default value = None)

    """
    trajectories = {}
    for episode in path.iterdir():
        trajectory_path = episode / "trajectory.npz"
        if not trajectory_path.exists():
            continue
        episode_data = np.load(episode / "trajectory.npz", allow_pickle=True)
        episode_data = dict(episode_data)
        if flatten:
            for key, value in episode_data.items():
                if mapping is not None:
                    if key in mapping:
                        key = mapping[key]
                if key == "time":
                    continue
                trajectories.setdefault(key, []).extend(value)
        else:
            if mapping is not None:
                for key, value in mapping.items():
                    episode_data[mapping[key]] = episode_data.pop(key)
            trajectories[episode.name] = episode_data
    if flatten:
        for key, value in trajectories.items():
            trajectories[key] = np.array(value)

    return trajectories


def cmd_train(args=None):
    from rl.utils import train, get_config
    from pathlib import Path
    import argparse
    import mergedeep

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="test")
    argparser.add_argument("--phantom", type=str, default="phantom3")
    argparser.add_argument("--target", type=str, default="bca")
    argparser.add_argument("--base-path", type=str, default="results")
    argparser.add_argument("--trial", type=str, default="1")
    argparser.add_argument("--n-runs", type=int, default=1)
    argparser.add_argument("--n-timesteps", type=int, default=int(6 * 10e4))
    args = argparser.parse_args()

    config_name = args.config
    config = get_config(config_name)

    mergedeep.merge(
        config,
        dict(
            task_kwargs=dict(target=args.target, phantom=args.phantom),
            train_kwargs=dict(time_steps=args.n_timesteps),
        ),
    )

    experiment_path = Path(f"{args.phantom}/{args.target}/{args.config}")
    print(f"Training {config_name} for {args.n_runs} runs with config:")
    __import__("pprint").pprint(config)
    train(
        experiment_path=experiment_path,
        target=args.target,
        phantom=args.phantom,
        config_name=args.config,
        base_path=Path.cwd() / args.base_path / args.trial,
        config=config,
        **config["train_kwargs"],
    )
