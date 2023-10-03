from cathsim.gym.envs import CathSim
from cathsim.gym.wrappers import MultiInputImageWrapper
from cathsim.rl.data import Trajectory
from cathsim.rl.evaluation import save_trajectories
from cathsim.rl.feature_extractors import CustomExtractor
from cathsim.rl import Config, make_gym_env
from cathsim.dm.visualization import point2pixel
from cathsim.dm.env import make_scene
from cathsim.dm.utils import filter_mask
from cathsim.dm.physics_functions import get_geom_pos as get_pos
from cathsim.dm.physics_functions import get_guidewire_geom_ids


from stable_baselines3 import SAC

import torch as th
import matplotlib.pyplot as plt
import numpy as np
import cv2

from shape_reconstruction import P_TOP, P_SIDE


import gymnasium as gym

from pathlib import Path


SCENE = make_scene([1, 2])


def get_policy(**kwargs):
    model_path = Path.cwd() / "models" / "sac.zip"
    # model = SAC.load(
    #     model_path,
    #     print_system_info=True,
    #     custom_objects={
    #         "policy_kwargs": dict(
    #             features_extractor_class=CustomExtractor,
    #         ),
    #     },
    #     **kwargs,
    # )
    return None


def get_env():
    env = gym.make(
        "cathsim/CathSim-v0",
        use_pixels=True,
        use_segment=True,
        random_init_distance=0.0,
        image_size=80,
        phantom="phantom3",
        target="bca",
    )
    env = MultiInputImageWrapper(
        env,
        grayscale=True,
    )
    return env


def process_observation(observation):
    pass


def get_images(env):
    physics = env.unwrapped.physics

    top = physics.render(480, 480, camera_id=0, scene_option=SCENE, segmentation=True)
    side = physics.render(480, 480, camera_id=2, scene_option=SCENE, segmentation=True)

    top = filter_mask(top)
    side = filter_mask(side)

    return top, side


def get_geom_pos(env):
    physics = env.unwrapped.physics
    geom_ids = get_guidewire_geom_ids(physics.model)
    geom_pos = get_pos(physics, geom_ids)

    return geom_pos


def plot_on_image(image, points, P, color=(0, 0, 255)):
    image = image.copy()
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)

    projected_points = point2pixel(points, P)
    for point in projected_points:
        if point[0] < 0 or point[1] < 0:
            continue
        if point[0] >= image.shape[0] or point[1] >= image.shape[0]:
            continue
        cv2.circle(image, tuple(point), 0, color, 3)
    return image


def visualize(top, side, geom_pos):
    top = top.copy()
    side = side.copy()
    geom_pos = geom_pos.copy()

    top = plot_on_image(top, geom_pos, P_TOP)
    side = plot_on_image(side, geom_pos, P_SIDE)

    combined = np.hstack((top, side))
    cv2.imshow("combined", combined)
    cv2.waitKey(1)


def save_data(step, top, side, actual):
    path = Path.cwd() / "data"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    plt.imsave((path / f"{step}_top.jpg").as_posix(), top, cmap="gray")
    plt.imsave((path / f"{step}_side.jpg").as_posix(), side, cmap="gray")
    np.save((path / f"{step}_actual.npy").as_posix(), actual)


def generate_data(n_samples: int = 50):
    path = Path.cwd() / "data"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    env = get_env()
    policy = get_policy()

    current_n_samples = 0
    while current_n_samples < n_samples:
        observation, _ = env.reset()
        done = False
        step = 0
        # while not done:
        for _ in range(50):
            # action, _ = policy(observation)
            action = [0.5, 0]
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(
                f"Step {current_n_samples} terminated: {terminated}, truncated: {truncated}, done: {done}",
                flush=True,
                end="\r",
            )
            top, side = get_images(env)
            geom_pos = get_geom_pos(env)
            geom_pos = np.array(geom_pos)
            save_data(current_n_samples, top, side, geom_pos)
            visualize(top, side, geom_pos)
            step += 1
            current_n_samples += 1


if __name__ == "__main__":
    generate_data()
