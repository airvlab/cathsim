from cathsim.gym.envs import CathSim
from cathsim.gym.wrappers import MultiInputImageWrapper
from cathsim.rl.feature_extractors import CustomExtractor
from cathsim.dm.visualization import point2pixel
from cathsim.dm.env import make_scene
from cathsim.dm.utils import filter_mask
from cathsim.dm.physics_functions import get_geom_pos as get_pos
from cathsim.dm.physics_functions import get_guidewire_geom_ids

import time

from stable_baselines3 import SAC

import matplotlib.pyplot as plt
import numpy as np
import cv2

from shape_reconstruction import P_TOP, P_SIDE


import gymnasium as gym

from pathlib import Path


SCENE = make_scene([1, 2])


def get_policy(**kwargs):
    model_path = Path.cwd() / "models" / "sac"
    model = SAC.load(
        model_path,
        print_system_info=True,
        custom_objects={
            "policy_kwargs": dict(
                features_extractor_class=CustomExtractor,
            ),
        },
        **kwargs,
    )
    return model.policy


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

    top_real = physics.render(480, 480, camera_id=0)
    side_real = physics.render(480, 480, camera_id=2)

    top_mask = physics.render(
        480, 480, camera_id=0, scene_option=SCENE, segmentation=True
    )
    side_mask = physics.render(
        480, 480, camera_id=2, scene_option=SCENE, segmentation=True
    )

    top_mask = filter_mask(top_mask)
    side_mask = filter_mask(side_mask)

    return top_real, side_real, top_mask, side_mask


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

    top_mask = plot_on_image(top, geom_pos, P_TOP)
    side_mask = plot_on_image(side, geom_pos, P_SIDE)

    combined_mask = np.hstack((top_mask, side_mask))
    cv2.imshow("combined", combined_mask)
    cv2.waitKey(1)


def save_data(path, step, top_real, side_real, top_mask, side_mask, actual):
    top_real = cv2.cvtColor(top_real, cv2.COLOR_RGB2GRAY)
    side_real = cv2.cvtColor(side_real, cv2.COLOR_RGB2GRAY)

    plt.imsave((path / f"{step}_top_real.jpg").as_posix(), top_real, cmap="gray")
    plt.imsave((path / f"{step}_side_real.jpg").as_posix(), side_real, cmap="gray")
    plt.imsave((path / f"{step}_top.jpg").as_posix(), top_mask, cmap="gray")
    plt.imsave((path / f"{step}_side.jpg").as_posix(), side_mask, cmap="gray")
    np.save((path / f"{step}_actual.npy").as_posix(), actual)


def save_data_simple(path, step, top_mask, side_mask, actual):
    plt.imsave((path / f"{step}_top.jpg").as_posix(), top_mask, cmap="gray")
    plt.imsave((path / f"{step}_side.jpg").as_posix(), side_mask, cmap="gray")
    np.save((path / f"{step}_actual.npy").as_posix(), actual)


def generate_data(n_samples: int = 500, resume: bool = False):
    path = Path.cwd() / "data_3"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    print(f"Generating data in {path.as_posix()}")

    env = get_env()
    policy = get_policy(env=env)

    current_n_samples = 0
    if resume:
        samples = list(path.glob("*.npy"))
        if len(samples) == 0:
            current_n_samples = 0
        else:
            samples = sorted(samples, key=lambda x: int(x.stem.split("_")[0]))
            current_n_samples = int(samples[-1].stem.split("_")[0])
            print(f"Resuming from {current_n_samples} samples")
    n_samples += current_n_samples
    while current_n_samples < n_samples:
        observation, _ = env.reset()
        done = False
        while not done:
            # action = [0.5, np.random.uniform(-1, 1)]
            action, _ = policy.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step {current_n_samples}", flush=True, end="\r")
            top_real, side_real, top_mask, side_mask = get_images(env)
            geom_pos = get_geom_pos(env)
            geom_pos = np.array(geom_pos)

            # save_data(path, current_n_samples, top_real, side_real, top_mask, side_mask, geom_pos)
            save_data_simple(path, current_n_samples, top_mask, side_mask, geom_pos)
            # visualize(top_mask, side_mask, geom_pos)
            current_n_samples += 1
            if done:
                break
    del env


if __name__ == "__main__":
    generate_data(resume=True)
