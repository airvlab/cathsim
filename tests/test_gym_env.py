# import pytest
# from unittest.mock import Mock
# from dm_control import composer
# from gym import spaces
# from cathsim.wrappers import DMEnvToGymWrapper  # replace with the path to your module
# from cathsim.cathsim.env_utils import make_dm_env
#
#
# class TestDMEnv:
#     @pytest.fixture
#     def env():
#         return make_dm_env()
#
#     def test_dm_env_to_gym_wrapper_init(env):
#         wrapper = DMEnvToGymWrapper(env)
#
#         # Check if control timestep is correctly passed
#         assert wrapper.metadata["video.frames_per_second"] == round(1.0 / 0.01)
#
#         # Check if image size is correctly passed
#         assert wrapper.image_size == 64
#
#         # Check the conversion of dm_control space to gym space for action and observation
#         assert isinstance(wrapper.action_space, spaces.Box)
#         assert isinstance(wrapper.observation_space, spaces.Box)
#
#         # The viewer should be None initially
#         assert wrapper.viewer is None
#
#     def test_dm_env_to_gym_wrapper_reset(mock_env):
#         wrapper = DMEnvToGymWrapper(mock_env)
#         obs = wrapper.reset()
#
#         # This just checks that reset() calls the correct methods and returns the correct value
#         # You might want to add more to this test depending on what you expect from reset()
#         mock_env.reset.assert_called_once()
#         assert obs == mock_env.reset.return_value.observation
#
#     def test_dm_env_to_gym_wrapper_render(mock_env):
#         wrapper = DMEnvToGymWrapper(mock_env)
#         img = wrapper.render()
#
#         # This just checks that render() calls the correct methods and returns the correct value
#         # You might want to add more to this test depending on what you expect from render()
#         mock_env.physics.render.assert_called_once_with(height=64, width=64, camera_id=0)
#         assert img == mock_env.physics.render.return_value
