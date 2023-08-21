import pytest
import numpy as np
from gym import spaces
from cathsim.rl.data import Trajectory
from toolz.dicttoolz import get_in

obs_cases = [
    # Single Box space
    spaces.Box(-1, 1, (1,)),
    # Nested Dict space with two Box spaces
    spaces.Dict({"obs1": spaces.Box(-1, 1, (1,)), "obs2": spaces.Box(-1, 1, (2,))}),
    # More deeply nested Dict space
    # spaces.Dict(
    #     {
    #         "obs1": spaces.Box(-1, 1, (1,)),
    #         "obs2": spaces.Dict(
    #             {
    #                 "obs2a": spaces.Box(-1, 1, (2,)),
    #                 "obs2b": spaces.Box(-1, 1, (3,)),
    #                 "obs2c": spaces.Dict({"obs2c1": spaces.Box(-1, 1, (4,))}),
    #             }
    #         ),
    #     }
    # ),
    # # Mixed space type
    # spaces.Dict({"obs1": spaces.Box(-1, 1, (1,)), "obs2": spaces.Discrete(5)}),
]
act_cases = [spaces.Box(-1, 1, (2, 1))]
info_cases = [spaces.Dict({"a": spaces.Box(-1, 1, (1,))})]


def sample_env(obs_space, act_space, info_space):
    obs = obs_space.sample()
    act = act_space.sample()
    info = info_space.sample()
    reward = np.random.random(1)
    return obs, act, reward, info


class TestTrajectory:
    @pytest.fixture(params=obs_cases)
    def obs_space(self, request):
        return request.param

    @pytest.fixture(params=act_cases)
    def act_space(self, request):
        return request.param

    @pytest.fixture(params=info_cases)
    def info_space(self, request):
        return request.param

    @pytest.fixture(params=[2, 3])
    def ep_len(self, request):
        return request.param

    @pytest.fixture
    def trajectory(self, obs_space, act_space, info_space, ep_len):
        trajectory = Trajectory()
        obs = sample_env(obs_space, act_space, info_space)[0]
        for i in range(ep_len):
            new_obs, act, reward, info = sample_env(obs_space, act_space, info_space)
            done = True if i == 1 else False
            trajectory.add_transition(
                obs=obs, act=act, reward=reward, info=info, done=done
            )
            obs = new_obs
            print("obs type", type(obs))
        return trajectory

    def test_init(self, ep_len, obs_space, act_space, info_space):
        trajectory = Trajectory()
        assert (
            trajectory.data is None
        ), f"Traj init with None but got: {trajectory.data}"
        obs, act, reward, info = sample_env(obs_space, act_space, info_space)
        trajectory = Trajectory(obs=obs, act=act, reward=reward, info=info)
        print(trajectory)

    def test_len_trajectory(self, trajectory, ep_len, obs_space, act_space, info_space):
        assert (
            len(trajectory) == ep_len
        ), f"traj length != ep_length: {len(trajectory)} != {ep_len}"
