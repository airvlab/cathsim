import pytest
from dm_control import composer
from cathsim.utils import make_dm_env
from cathsim import Phantom


class TestDMEnv:

    @pytest.fixture
    def phantom(self):
        return Phantom('phantom3.xml')

    @pytest.fixture
    def env(self):
        return make_dm_env()

    @pytest.mark.parametrize('target', ['bca', 'lcca'])
    def test_target(self, target, phantom):
        env = make_dm_env(target=target)
        assert isinstance(env, composer.Environment)
        env_target = env.task.target_pos
        phantom_sites = phantom.sites
        assert (env_target == phantom_sites[target]).all()

    def test_can_compile(self, env):
        def random_policy(time_step):
            del time_step
            return [0, 0]

        time_step = env.reset()
        action = random_policy(time_step)
        time_step = env.step(action)
        assert time_step.reward is not None
