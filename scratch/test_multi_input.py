from cathsim.utils import make_env
# from stable_baselines3.common.env_checker import check_env
from gym.utils.env_checker import check_env

wrapper_kwargs = dict(
    # time_limit=300,
    use_pixels=True,
    grayscale=True,
    resize_shape=80,
)

algo_kwargs = dict(
    policy='MultiInputPolicy',
)

env_kwargs = dict(
)

env = make_env(
    wrapper_kwargs=wrapper_kwargs,
)

print(env.observation_space.keys())
print(env.observation_space.spaces['pixels'].shape)
obs = env.reset()
print(obs['pixels'].shape)

check_env(env)
