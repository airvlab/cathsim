import cathsim
import cv2
import gymnasium as gym
import gymnasium.envs.mujoco.mujoco_env
import gymnasium.envs.mujoco.swimmer_v4
import pytest
from gymnasium.utils.env_checker import check_env

env = gym.make("CathSim-v1")
check_env(env.unwrapped, skip_render_check=True)
exit()

ob, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    ob, reward, terminated, _, info = env.step(action)
    # print(f"Reward: {reward}, Info: {info}")
    cv2.imshow("Top Camera", ob["pixels"])
    cv2.waitKey(1)
