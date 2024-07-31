from gymnasium.envs.registration import register

register(
    "CathSim-v1",
    "cathsim.envs:CathSim",
    max_episode_steps=300,
    nondeterministic=True,
)
