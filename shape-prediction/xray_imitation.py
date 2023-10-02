import cv2
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from cathsim.gym.envs import CathSim
from cathsim.gym.wrappers import TransformDictObservation


def image_augmentation(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.GaussianBlur(observation, (5, 5), 0)

    std_dev = np.sqrt(0.002)
    noise = np.random.normal(0, std_dev, observation.shape).astype(observation.dtype)
    observation = cv2.add(observation, noise)

    observation = np.clip(observation, 0, 255)

    return observation


def guidewire_augmentation(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.GaussianBlur(observation, (5, 5), 0)

    # Add Gaussian white noise with zero mean and 0.002 variance
    std_dev = np.sqrt(0.002)
    noise = np.random.normal(0, std_dev, observation.shape).astype(observation.dtype)
    observation = cv2.add(observation, noise)

    observation = np.clip(observation, 0, 255)

    return observation


if __name__ == "__main__":
    env = gym.make(
        "cathsim/CathSim-v0", use_pixels=True, image_size=480, use_segment=True
    )
    env = TransformDictObservation(env, image_augmentation, "pixels")
    obs, _ = env.reset()
    plt.imshow(obs["pixels"], cmap="gray")
    plt.axis("off")
    plt.show()
    env.unwrapped.print_spaces()
