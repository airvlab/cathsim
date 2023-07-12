import os
import torch as th
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, HerReplayBuffer

from dm_control.viewer.application import Application
from imitation.algorithms import bc
from stable_baselines3.common import policies
from cathsim.wrappers import Dict2Array
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym
from cathsim.cathsim.env_utils import make_gym_env
from abc import ABC, abstractmethod


def filter_mask(segment_image: np.ndarray):
    geom_ids = segment_image[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    segment_image = 255 * geom_ids
    return segment_image


class CnnPolicy(policies.ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


ALGOS = {
    'ppo': PPO,
    'sac': SAC,
    'bc': bc,
}

RESULTS_PATH = Path(__file__).parent.parent / 'results'
EXPERIMENT_PATH = RESULTS_PATH / 'experiments'
EVALUATION_PATH = RESULTS_PATH / 'evaluation'


def make_experiment(experiment_path: Path = None) -> tuple:
    """Create experiment directory structure.

    experiment_path: Path to experiment directory

    returns:
        model_path: Path to save models
        log_path: Path to save logs
        eval_path: Path to save evaluation results

    example:
        model_path, log_path, eval_path = make_experiment('test')
    """
    assert experiment_path, 'experiment_path must be specified'
    experiment_path = EXPERIMENT_PATH / experiment_path
    model_path = experiment_path / 'models'
    eval_path = experiment_path / 'eval'
    log_path = experiment_path / 'logs'
    for dir in [experiment_path, model_path, log_path, eval_path]:
        dir.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def make_env(
    n_envs: int = 1,
    config: dict = {},
    monitor_wrapper=True,
    **kwargs
):

    def _init() -> gym.Env:
        env = make_gym_env(config=config)
        return env

    if n_envs == 1:
        env = DummyVecEnv([_init])
    else:
        env = SubprocVecEnv([_init for _ in range(n_envs)])

    if monitor_wrapper:
        if n_envs == 1:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env)
        else:
            from stable_baselines3.common.vec_env import VecMonitor
            env = VecMonitor(env)

    return env


def train(algo: str,
          experiment: str,
          experiment_path: Path = None,
          n_runs: int = 4,
          time_steps: int = 500_000,
          evaluate: bool = False,
          device: str = None,
          n_envs: int = None,
          vec_env: bool = True,
          config: dict = {},
          **kwargs):
    from rl.evaluation import evaluate_policy
    algo_kwargs = config.get('algo_kwargs', {})

    if not device:
        device = 'cuda' if th.cuda.is_available() else 'cpu'
    n_envs = n_envs or os.cpu_count() // 2

    assert algo in ALGOS.keys(), f'algo must be one of {ALGOS.keys()}'
    assert experiment_path, 'experiment_path must be specified'
    assert n_runs > 0, 'n_runs must be greater than 0'
    assert time_steps > 0, 'time_steps must be greater than 0'
    assert device in ['cpu', 'cuda'], 'device must be one of [cpu, cuda]'
    assert n_envs > 0, 'n_envs must be greater than 0'

    experiment_path = experiment_path / experiment
    model_path, log_path, eval_path = make_experiment(experiment_path)

    n_envs = n_envs or os.cpu_count() // 2

    env = make_env(n_envs=n_envs, config=config)

    for seed in range(n_runs):
        if (model_path / f'{algo}_{seed}.zip').exists():
            print(f'Model {algo} {seed} already exists, skipping')
            pass
        else:
            for key, value in algo_kwargs.items():
                __import__('pprint').pprint(f'{key}: {value}')
            model = ALGOS[algo](algo_kwargs.get('policy', 'MlpPolicy'),
                                env,
                                device=device,
                                verbose=1,
                                tensorboard_log=log_path,
                                policy_kwargs=algo_kwargs.get('policy_kwargs', {}),
                                **kwargs)

            model.learn(total_timesteps=time_steps,
                        log_interval=10,
                        tb_log_name=f'{algo}_{seed}',
                        progress_bar=True,
                        reset_num_timesteps=False)

            model.save(model_path / f'{algo}_{seed}.zip')

            if evaluate:
                results = evaluate_policy(model, env, 10)
                np.savez_compressed(eval_path / f'{algo}_{seed}', **results)
            th.cuda.empty_cache()


def cmd_visualize_agent(args=None):
    import cv2
    from cathsim.cathsim import make_env
    import argparse as ap
    from scratch.bc.custom_networks import CustomPolicy
    parser = ap.ArgumentParser()
    parser.add_argument('--config', type=str, default='full')
    parser.add_argument('--phantom', type=str, default='phantom3')
    parser.add_argument('--target', type=str, default='bca')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-video', type=bool, default=False)
    parser.add_argument('--get_images', type=bool, default=False)
    parser.add_argument('--algo', type=str, default='sac')
    parser.add_argument('--visualize-sites', type=bool, default=False)
    args = parser.parse_args()

    algo = args.algo

    if args.save_video:
        import moviepy.editor as mpy

    path = Path(f'{args.phantom}/{args.target}/{args.config}')
    config = get_config(args.config)
    model_path, log_path, eval_path = make_experiment(path)

    if args.seed is None:
        model_path = model_path / algo
    else:
        model_path = model_path / (algo + f'_{args.seed}.zip')
    video_path = model_path.parent.parent / 'videos'
    images_path = model_path.parent.parent / 'images'

    assert model_path.exists(), f'{model_path} does not exist'

    # if config == 'pixels':
    #     config['wrapper_kwargs']['get_images'] = args.get_images
    config['task_kwargs']['target'] = args.target
    config['task_kwargs']['phantom'] = args.phantom
    config['task_kwargs']['visualize_sites'] = args.visualize_sites

    if algo == 'bc':
        config['wrapper_kwargs']['channel_first'] = True
        env = make_env(config=config)
        env = Dict2Array(env)
        model = CnnPolicy(observation_space=env.observation_space,
                          action_space=env.action_space,
                          lr_schedule=lambda _: th.finfo(th.float32).max,
                          ).load(model_path)
    else:
        env = make_env(config=config)
        model = ALGOS[algo].load(model_path)

    for episode in range(1):
        obs = env.reset()
        done = False
        frames = []
        segment_frames = []
        rewards = []
        # scene_option = make_scene(geom_groups=[1, 2])
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)

            image = env.render('rgb_array', image_size=480)
            # segment_image = env.env.env.env.physics.render(480, 480, camera_id=0, scene_option=scene_option,
            #                                                segmentation=True)
            # segment_image = filter_mask(segment_image)
            # segment_frames.append(segment_image)
            # print(segment_image.shape)
            frames.append(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('segment', segment_image)
            cv2.imshow('image', image)
            cv2.waitKey(1)

        if args.save_video:
            os.makedirs(video_path, exist_ok=True)
            frames = [mpy.ImageClip(m).set_duration(0.1)for m in frames]
            clip = mpy.concatenate_videoclips(frames, method='compose')
            clip.write_videofile(
                (video_path / f'{algo}_{episode}.mp4').as_posix(), fps=60)
        if args.get_images:
            import matplotlib.pyplot as plt
            os.makedirs(images_path, exist_ok=True)
            print(f'Saving images to {images_path}')
            for i in range(len(frames)):
                plt.imsave(f'{images_path}/{algo}_{episode}_{i}.png', frames[i])
                plt.imsave(f'{images_path}/{algo}_{episode}_{i}_segment.png', segment_frames[i], cmap='gray')


class Application(Application):

    def __init__(self, title, width, height, trial_path=None):
        super().__init__(title, width, height)
        from dm_control.viewer import user_input

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)
        self.null_action = np.zeros(2)
        self._step = 0
        self._episode = 0
        if trial_path.exists():
            episode_paths = sorted(trial_path.iterdir())
            if len(episode_paths) > 0:
                episode_num = episode_paths[-1].name.split('_')[-1]
                self._episode = int(episode_num) + 1
        self._policy = None
        self._trajectory = {}
        self._trial_path = trial_path
        self._episode_path = self._trial_path / f'episode_{self._episode:02}'
        self._images_path = self._episode_path / 'images'
        self._images_path.mkdir(parents=True, exist_ok=True)

    def _save_transition(self, observation, action):
        for key, value in observation.items():
            if key != 'top_camera':
                self._trajectory.setdefault(key, []).append(value)
            else:
                image_path = self._images_path / f'{self._step:03}.png'
                plt.imsave(image_path.as_posix(), value)
        self._trajectory.setdefault('action', []).append(action)

    def _initialize_episode(self):
        trajectory_path = self._episode_path / 'trajectory'
        np.savez_compressed(trajectory_path.as_posix(), **self._trajectory)
        self._restart_runtime()
        print(f'Episode {self._episode:02} finished')
        self._trajectory = {}
        self._step = 0
        self._episode += 1
        # change the episode path to the new episode
        self._episode_path = self._trial_path / f'episode_{self._episode:02}'
        self._images_path = self._episode_path / 'images'
        self._images_path.mkdir(parents=True, exist_ok=True)

    def perform_action(self):
        print(f'step {self._step:03}')
        time_step = self._runtime._time_step
        if not time_step.last():
            self._advance_simulation()
            action = self._runtime._last_action
            self._save_transition(time_step.observation, action)
            self._step += 1
        else:
            self._initialize_episode()

    def _move_forward(self):
        self._runtime._default_action = [1, 0]
        self.perform_action()

    def _move_back(self):
        self._runtime._default_action = [-1, 0]
        self.perform_action()

    def _move_left(self):
        self._runtime._default_action = [0, -1]
        self.perform_action()

    def _move_right(self):
        self._runtime._default_action = [0, 1]
        self.perform_action()


def launch(environment_loader, policy=None, title='Explorer', width=1024,
           height=768, trial_path=None):
    app = Application(title=title, width=width, height=height,
                      trial_path=trial_path)
    app.launch(environment_loader=environment_loader, policy=policy)


def record_expert_trajectories(trial_name: Path):
    from cathsim.cathsim.env import Tip, Guidewire, Navigate
    from cathsim.cathsim.phantom import Phantom
    from dm_control import composer

    trial_path = Path(__file__).parent.parent / 'expert' / trial_name
    try:
        trial_path.mkdir(parents=True)
    except FileExistsError:
        cont_training = input(
            f'Trial {trial_name} already exists. Continue? [y/N] ')
        cont_training = 'n' if cont_training == '' else cont_training
        if cont_training.lower() == 'y':
            pass
        else:
            print('Aborting')
            exit()

    phantom = Phantom()
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_image=True,
    )
    env = composer.Environment(
        task=task,
        time_limit=200,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    action_spec_name = '\t' + env.action_spec().name.replace('\t', '\n\t')
    print('\nAction Spec:\n', action_spec_name)
    time_step = env.reset()
    print('\nObservation Spec:')
    for key, value in time_step.observation.items():
        print('\t', key, value.shape)

    launch(env, trial_path=trial_path)


def cmd_record_traj(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--trial_name', type=str, default='test')
    args = parser.parse_args(args)
    record_expert_trajectories(Path(args.trial_name))


def get_config(config_name):
    import yaml
    from mergedeep import merge
    from rl.custom_extractor import CustomExtractor

    configs_path = Path(__file__).parent / 'config'
    main_config_path = configs_path / 'main.yaml'
    config_path = configs_path / (config_name + '.yaml')
    main_config = yaml.safe_load(open(main_config_path, 'r'))
    config = yaml.safe_load(open(config_path, 'r'))
    config = merge(main_config, config)

    policy_kwargs = main_config['algo_kwargs'].get('policy_kwargs', {})
    feature_extractor_class = policy_kwargs.get('features_extractor_class', None)
    if feature_extractor_class == 'CustomExtractor':
        main_config['algo_kwargs']['policy_kwargs']['features_extractor_class'] = CustomExtractor

    if main_config['algo_kwargs']['replay_buffer_class'] == 'HerReplayBuffer':
        main_config['algo_kwargs']['replay_buffer_class'] = HerReplayBuffer
    return main_config


def process_human_trajectories(path: Path, flatten=False, mapping: dict = None):
    trajectories = {}
    for episode in path.iterdir():
        trajectory_path = episode / 'trajectory.npz'
        if not trajectory_path.exists():
            continue
        episode_data = np.load(episode / 'trajectory.npz', allow_pickle=True)
        episode_data = dict(episode_data)
        if flatten:
            for key, value in episode_data.items():
                if mapping is not None:
                    if key in mapping:
                        key = mapping[key]
                if key == 'time':
                    continue
                trajectories.setdefault(key, []).extend(value)
        else:
            if mapping is not None:
                for key, value in mapping.items():
                    episode_data[mapping[key]] = episode_data.pop(key)
            trajectories[episode.name] = episode_data
    if flatten:
        for key, value in trajectories.items():
            trajectories[key] = np.array(value)

    return trajectories


if __name__ == '__main__':

    exit()

    # env = make_env(
    #     wrapper_kwargs=wrapper_kwargs,
    #     task_kwargs=task_kwargs
    # )
    # experiment_name = 'simple'
    # model_path = EXPERIMENTS_PATH / experiment_name / 'models' / 'sac_0.zip'
    # model = SAC.load(model_path, env=env, **algo_kwargs)
    # evaluate_policy(model, env, experiment_name, n_episodes=30)
    # summary_results = analyze_model(f'{experiment_name}.npz')
    # evaluate_all()
    # __import__('pprint').pprint(summary_results)
