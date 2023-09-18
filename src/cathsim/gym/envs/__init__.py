from cathsim.gym.envs.cathsim import CathSim
from gymnasium.envs.registration import register


register(
    id="cathsim/CathSim-v0",
    entry_point="cathsim.gym.envs:CathSim",
    max_episode_steps=300,
    nondeterministic=True,
)
