import numpy as np
import torch as th
from pathlib import Path

from rl.sb3.evaluation import evaluate_policy
from rl.imitation.utils import filter
from rl.sb3.sb3_utils import make_experiment, get_config
from cathsim.wrappers import Dict2Array

from cathsim.utils import make_env

from imitation.data.types import Transitions

from scratch.bc.custom_networks import CustomPolicy
from scratch.bc import bc


def info_filter(info):
    info = info.pop('features')
    info = np.squeeze(info)
    return info


def obs_filter(obs):
    obs = obs.pop('pixels')
    obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    return obs


phantom = 'phantom3'
target = 'bca'
experiment_name = 'bc'

if __name__ == "__main__":
    path = Path(f'{phantom}/{target}/{experiment_name}')
    model_path, log_path, eval_path = make_experiment(path)

    rng = np.random.default_rng(0)
    transitions = np.load('./rl/imitation/trajectories/bca.npz', allow_pickle=True)
    transitions = filter(info_filter, 'infos', transitions)
    transitions = filter(obs_filter, ['obs', 'next_obs'], transitions)
    transitions = {k: np.array(v) for k, v in transitions.items() if k != 'reward'}
    transitions = Transitions(**dict(transitions))

    config = get_config('pixels')
    config['wrapper_kwargs']['channel_first'] = True
    env = make_env(config)
    env = Dict2Array(env)
    print(env.observation_space.shape)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=CustomPolicy(observation_space=env.observation_space,
                            action_space=env.action_space,
                            lr_schedule=lambda _: th.finfo(th.float32).max,
                            ),
    )

    bc_trainer.train(n_epochs=800)
    bc_trainer.policy.save(model_path / 'bc')
    results = evaluate_policy(bc_trainer.policy, env, n_episodes=10)
    np.savez_compressed(eval_path / 'bc.npz', **results)
    for key, value in results.items():
        print(f"Episode {key}: {len(value['forces'])} steps")
