import gymnasium as gym
from cathsim.gym.envs import CathSim, make_gym_env

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from cathsim.rl.utils import Config
import os


if __name__ == "__main__":
    config = Config("full")
    env = make_gym_env(config, n_envs=os.cpu_count() // 2, monitor_wrapper=True)

    for k, v in env.observation_space.spaces.items():
        print(k, v)

    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(reward)
        # vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()
