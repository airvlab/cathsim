import numpy as np
import os
import torch as th
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
from scratch.force_prediction.force_prediction_utils import ForcePrediction, SimpleForcePrediction, MeanForcePrediction, SELUActivation
from scratch.force_prediction.force_prediction_utils import TransitionsDataset, clean_transitions


th.manual_seed(42)
n_k = [1e3, 1e4, 1e5]
target = 'bca'
# CONSTANTS
num_cpu = os.cpu_count()
seed = th.Generator().manual_seed(42)

MODELS = [SELUActivation]


transitions_path = Path.cwd() / 'rl' / 'imitation' / 'trajectories' / f'phantom3_{target}_{1e5}.npz'
transitions = np.load(transitions_path, allow_pickle=True)
transitions = clean_transitions(transitions, filter_keys=['forces', 'obs', 'features'])


# transitions_lcca = np.load(Path.cwd() / 'rl' / 'imitation' / 'trajectories' / 'phantom3_lcca_100000.npz', allow_pickle=True)
# transitions_lcca = clean_transitions(transitions_lcca, filter_keys=['forces', 'obs', 'features'])


feature_dim = transitions['features'].shape[0]

dataset = TransitionsDataset(transitions, train=True)
test_dataset = TransitionsDataset(transitions, train=False)


train_set_size = int(0.8 * len(dataset))
val_set_size = len(dataset) - train_set_size
train_set, val_set = th.utils.data.random_split(dataset, [train_set_size, val_set_size], generator=seed)

train_loader = DataLoader(train_set, num_workers=num_cpu)
val_loader = DataLoader(val_set, num_workers=num_cpu)
test_loader = DataLoader(test_dataset)

test_trainer = pl.Trainer(logger=False)

if __name__ == "__main__":
    experiment = 'force_pred_k'
    for k in n_k:
        # get only the first k transitions
        sub_train_set = th.utils.data.Subset(train_set, range(int(k * .8 ** 2)))
        train_loader = DataLoader(sub_train_set, num_workers=num_cpu)

        model = SELUActivation(input_channels=1, features_dim=768, output_shape=1)
        log_dir = Path.cwd() / 'scratch' / 'force_prediction' / 'lightning_logs' / experiment

        trainer = pl.Trainer(max_epochs=30, default_root_dir=log_dir,
                             logger=pl.loggers.TensorBoardLogger(log_dir, name=f'{target}_k_{k}'))

        trainer.fit(model, train_loader, val_loader)
        test_trainer.test(model, test_loader)
