import cathsim
import cv2
import gymnasium as gym

env = gym.make("CathSim-v1")


ob, info = env.reset()
for _ in range(1000):
    # action = cathsim.action_space.sample()
    action = [0.1, 0]
    ob, reward, terminated, _, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    cv2.imshow("Top Camera", ob["pixels"])
    cv2.waitKey(1)
