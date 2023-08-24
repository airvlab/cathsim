from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import GrayScaleObservation
from cathsim.gym.envs import CathSim
import gymnasium as gym
import cv2

env = gym.make("cathsim/CathSim-v0", target="bca")
# env = GrayScaleObservation(env, keep_dim=True)
check_env(env, warn=False, skip_render_check=True)

for i in range(10):
    obs = env.reset()
    done = False
    while not done:
        action = [1, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        image = env.render_frame(image_size=480)
        cv2.imshow("image", image)
        cv2.waitKey(1)
