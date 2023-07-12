from rl.utils import train, get_config
from pathlib import Path
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, default='test')
argparser.add_argument('--experiment-path', type=str, default='test')
argparser.add_argument('--target', type=str, default='bca')
argparser.add_argument('--phantom', type=str, default='phantom3')
argparser.add_argument('--n-runs', type=int, default=1)
argparser.add_argument('--n-timesteps', type=int, default=6 * 10e4)
args = argparser.parse_args()

config_name = args.config
config = get_config(config_name)
config['task_kwargs']['target'] = args.target
config['task_kwargs']['phantom'] = args.phantom
config['train_kwargs']['time_steps'] = args.n_timesteps

if args.experiment_path is None:
    experiment_path = Path(f'{args.phantom}/{args.target}')
else:
    experiment_path = Path(args.experiment_path)

config['train_kwargs'].pop('n_runs')

if __name__ == "__main__":
    print('Training {} for {} runs with config:'.format(config_name, args.n_runs))
    __import__('pprint').pprint(config)
    train(
        experiment_path=experiment_path,
        experiment=args.config,
        n_runs=args.n_runs,
        config=config,
        **config['train_kwargs']
    )
