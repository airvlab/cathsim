from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from rl.feature_extractors import CustomExtractor
from rl.sb3.sb3_utils import get_config, make_experiment, make_vec_env
from cathsim.cathsim.env_utils import make_env
from pathlib import Path
import os

if __name__ == "__main__":
    phantom = 'phantom3'
    target = 'bca'
    path = Path(f'{phantom}/{target}/her')

    model_path, log_path, eval_path = make_experiment(path)

    config = get_config('internal')
    config['task_kwargs']['phantom'] = phantom
    config['task_kwargs']['target'] = target
    config['wrapper_kwargs']['goal_env'] = True

    n_cpu = os.cpu_count()
    env = make_vec_env(n_cpu, config)
    # env = make_dm_env(**config['task_kwargs'])
    # env = DMEnvToGymWrapper(env)
    # env = GoalEnvWrapper(env)
    # env = Monitor(env)
    # env = Dict2Array(env)
    # env = MultiInputImageWrapper(env, grayscale=True)

    obs = env.reset()
    print('\nReset Observation Space:')
    for key, value in obs.items():
        print("\t", key, value.shape, value.dtype)
    print('\n')

    model = DDPG(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        device='cuda',
        learning_starts=n_cpu * 400,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=dict(
            features_extractor_class=CustomExtractor,
        ),
        seed=0,
    )
    model.learn(600_000, progress_bar=True, log_interval=10, tb_log_name='her_sac')
    model.save(model_path / 'her_sac.zip')