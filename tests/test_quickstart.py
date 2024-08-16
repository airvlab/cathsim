import cathsim.gym.envs
import gymnasium as gym

task_kwargs = dict(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom="phantom3",
    target="bca",
)

env = gym.make("cathsim/CathSim-v0", **task_kwargs)


obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for obs_key in obs:
        print(obs_key, obs[obs_key].shape)
    print(reward)
    print(terminated)
    print(truncated)
    for info_key in info:
        print(info_key, info[info_key])
