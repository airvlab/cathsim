import gym
from cathsim.rl.data import Trajectory, TrajectoriesDataset
from torch.utils import data


def test_trajectory(
    obs_space: gym.spaces, act_space: gym.spaces, len: int = 3, n_trajectories: int = 2
):
    trajectories = []
    obs = obs_space.sample()
    act = act_space.sample()

    for n in range(n_trajectories):
        trajectory = Trajectory(keys=["obs", "act"])
        for i in range(len):
            trajectory.add_transition(obs=obs, act=act)
        trajectories.append(trajectory)

    t_dataset = TrajectoriesDataset(trajectories)

    return trajectories
