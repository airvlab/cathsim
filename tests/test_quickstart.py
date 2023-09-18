# from cathsim.utils import make_dm_env
# from cathsim.gym.wrappers import DMEnvToGymWrapper
#
# env = make_dm_env(
#     dense_reward=True,
#     success_reward=10.0,
#     delta=0.004,
#     use_pixels=False,
#     use_segment=False,
#     image_size=64,
#     phantom="phantom3",
#     target="bca",
# )
#
# env = DMEnvToGymWrapper(env)
#
# obs = env.reset()
# for _ in range(1):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     for obs_key in obs:
#         print(obs_key, obs[obs_key].shape)
#     print(reward)
#     print(done)
#     for info_key in info:
#         print(info_key, info[info_key])
