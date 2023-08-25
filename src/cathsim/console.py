import os
import torch as th
from pathlib import Path
import argparse as ap

from cathsim.rl.utils import ALGOS, generate_experiment_paths
from cathsim.rl import Config


def cmd_visualize_agent(args=None):
    """Visualize a trained agent.

    :param args:  (Default value = None)

    """
    import cv2
    from cathsim.utils import make_gym_env
    from cathsim.visualization import point2pixel
    import argparse as ap
    from cathsim.wrappers import SingleDict2Array
    from stable_baselines3.common.policies import ActorCriticCnnPolicy

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
    config = Config(args.config)
    model_path, log_path, eval_path = generate_experiment_paths(
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
        env = SingleDict2Array(env)
        model = ActorCriticCnnPolicy(
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


def cmd_run_env(args=None):
    """
    Runs the environment.

    :param args:  (Default value = None)

    """
    from argparse import ArgumentParser
    from dm_control import composer
    from cathsim import Phantom, Guidewire, Tip, Navigate
    from cathsim.utils import launch
    import numpy as np

    ap = ArgumentParser()
    ap.add_argument("--interact", type=bool, default=True)
    ap.add_argument("--phantom", default="phantom3", type=str)
    ap.add_argument("--target", default="bca", type=str)
    ap.add_argument("--image_size", default=80, type=int)
    ap.add_argument("--visualize-target", action="store_true")

    args = ap.parse_args(args)

    task = Navigate(
        phantom=Phantom(args.phantom + ".xml"),
        guidewire=Guidewire(),
        tip=Tip(),
        use_pixels=True,
        use_segment=True,
        target=args.target,
        visualize_sites=False,
        visualize_target=args.visualize_target,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    launch(env)


def cmd_train(args=None):
    from cathsim.rl import train

    parser = ap.ArgumentParser()
    parser.add_argument("-a", "--algo", type=str, default="sac")
    parser.add_argument("-c", "--config", type=str, default="test")
    parser.add_argument("-t", "--target", type=str, default="bca")
    parser.add_argument("-p", "--phantom", type=str, default="phantom3")
    parser.add_argument("--trial-name", type=str, default="test-trial")
    parser.add_argument("--base-path", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--n-timesteps", type=int, default=int(6e5))
    parser.add_argument("-e", action="store_true")
    args = parser.parse_args()

    train(
        algo=args.algo,
        config_name=args.config,
        target=args.target,
        phantom=args.phantom,
        trial_name=args.trial_name,
        base_path=Path.cwd() / args.base_path,
        n_timesteps=args.n_timesteps,
        n_runs=args.n_runs,
        evaluate=args.e,
    )
