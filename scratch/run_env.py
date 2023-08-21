
def run_env(args=None):
    from argparse import ArgumentParser
    from dm_control.viewer import launch

    parser = ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=80)
    parser.add_argument('--tip_n_bodies', type=int, default=4)
    parser.add_argument('--interact', type=bool, default=True)
    target = 'bca'

    parsed_args = parser.parse_args(args)

    phantom = Phantom()
    tip = Tip(n_bodies=parsed_args.tip_n_bodies)
    guidewire = Guidewire(n_bodies=parsed_args.n_bodies)

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target=target,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    physics = env.physics
    print(phantom.sites[target])
    print(env._task._target_pos)
    # print(env._task.get_head_pos(physics))
    assert (phantom.sites[target] == env._task._target_pos).all(), \
        f"{phantom.sites[target]} != {env._task._target_pos}"
    assert (env._task.get_reward(physics) == env._task.compute_reward(
        env._task.get_head_pos(physics), env._task._target_pos)), \
        f"{env._task.get_reward(physics)} != {env._task.compute_reward(env._task.get_head_pos(physics), env._task._target_pos)}"
    # exit()

    # print(env._task.get_contact_forces(physics))
    #
    # def random_policy(time_step):
    #     del time_step  # Unused.
    #     return [0, 0]
    #
    # launch(env, policy=random_policy)
    # exit()

    # Launch the viewer application.
    # if parsed_args.interact:
    #     from cathsim.utils import launch
    #     launch(env)
    # else:
    #     launch(env, policy=random_policy)

    # camera = mujoco.Camera(env.physics, 480, 480, 0)
    # print(camera.matrix)
    # exit()

    # env.reset()
    #
    # for k, v in env.observation_spec().items():
    #     print(k, v.dtype, v.shape)
    #
    # def plot_obs(obs):
    #     import matplotlib.pyplot as plt
    #     import cv2
    #     top_camera = obs['pixels']
    #     guidewire = obs['guidewire']
    #     # phantom = obs['phantom']
    #     # top_camera = cv2.cvtColor(top_camera, cv2.COLOR_RGB2GRAY)
    #
    #     # plot the phantom and guidewire in subplot
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(top_camera)
    #     ax[0].axis('off')
    #     ax[0].set_title('top_camera')
    #     ax[1].imshow(guidewire)
    #     ax[1].set_title('guidewire segmentation')
    #     ax[1].axis('off')
    #     # ax[2].imshow(phantom)
    #     # ax[2].set_title('phantom segmentation')
    #     # ax[2].axis('off')
    #     plt.show()
    #     # exit()
    #
    #     # plt.imsave('./figures/phantom_mask.png', np.squeeze(phantom))
    #     plt.imsave('./figures/phantom_2.png', top_camera)
    #     # cv2.imwrite('./figures/phantom.png', top_camera)
    #     exit()
    # env.task.camera_matrix = env.task.get_camera_matrix(physics)
    for i in range(100):
        action = np.zeros(env.action_spec().shape)
        action[0] = 1
        timestep = env.step(action)
        image = timestep.observation['pixels']
        # overlap the contact forces on the image with the color based on the magnitude
        forces = env.task.get_contact_forces(env.physics)
        max_force = 6
        min_force = 0
        for i in range(len(forces['pos'])):
            cv2.circle(image, tuple(forces['pos'][i]), 2, (0, int((forces['force'][i] - min_force) / (max_force - min_force) * 255), 0), -1)
        cv2.imshow('top_camera', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        # print(env
        # print(env.task.get_contact_forces(physics))


