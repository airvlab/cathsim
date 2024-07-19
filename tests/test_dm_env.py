import pytest
from cathsim.dm import make_dm_env

env = make_dm_env(
    phantom="phantom3",
    use_pixels=True,
    use_segment=True,
    target="bca",
    image_size=80,
    visualize_sites=False,
    visualize_target=True,
    sample_target=True,
    target_from_sites=False,
)

env._task.get_guidewire_geom_pos(env.physics)
print(env._task.get_camera_matrix(camera_name="top_camera", image_size=80))


def random_policy(time_step):
    del time_step  # Unused
    return [0, 0]


# loop 2 episodes of 2 steps
for episode in range(2):
    time_step = env.reset()
    for step in range(2):
        action = random_policy(time_step)
        time_step = env.step(action)
        img = env.physics.render(height=480, width=480, camera_id=0)
        contact_forces = env._task.get_contact_forces(env.physics, threshold=0.01)
        print(contact_forces)
        print(env._task.get_head_pos(env._physics))
        print(env._task.target_pos)
        print("Reward", time_step.reward)
