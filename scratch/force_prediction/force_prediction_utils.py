from collections import defaultdict
from pathlib import Path
import numpy as np
import math
import os

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

from rl.imitation.utils import filter
import pytorch_lightning as pl


def lecun_normal_(tensor: th.Tensor) -> th.Tensor:
    input_size = tensor.shape[-1]  # Assuming that the weights' input dimension is the last.
    std = math.sqrt(1 / input_size)
    with th.no_grad():
        return tensor.normal_(-std, std)


def info_filter(info):
    info = {k: np.array(v) for k, v in info.items() if k in ['features', 'forces']}
    info['features'] = info['features'].squeeze()

    return info


def shuffle_transitions(transitions: dict) -> dict:
    """Shuffle transitions dict"""
    indices = np.arange(len(transitions['obs']))
    np.random.shuffle(indices)
    shuffled_transitions = {}
    for key, value in transitions.items():
        if isinstance(value, np.ndarray):
            shuffled_transitions[key] = value[indices]
        else:
            shuffled_transitions[key] = [value[i] for i in indices]
    return shuffled_transitions


def obs_filter(obs):
    """Filter observations by getting the pixels (image) and making it channel first"""
    obs = obs.pop('pixels')
    obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    return obs


def group_list_of_dicts(list_of_dicts: list) -> dict:
    """Group elements of a list of dicts into a single dict"""
    grouped_dict = defaultdict(list)

    for i, d in enumerate(list_of_dicts):
        for key, value in d.items():
            grouped_dict[key].append(value)
    return grouped_dict


def clean_transitions(transitions: dict, filter_keys: list = ['infos', 'obs', 'forces']):
    """Clean transitions dict by removing unwanted keys and grouping infos"""
    transitions = filter(info_filter, 'infos', transitions)
    transitions = filter(obs_filter, ['obs', 'next_obs'], transitions)
    infos = group_list_of_dicts(transitions.pop('infos'))
    new_transitions = {}
    for key, value in infos.items():
        transitions[key] = value
    for key, value in transitions.items():
        if key in filter_keys:
            new_transitions[key] = value
    new_transitions = shuffle_transitions(transitions)
    for key, value in new_transitions.items():
        new_transitions[key] = np.array(value)
    return new_transitions


class TransitionsDataset(Dataset):
    def __init__(self, transitions, transform_image=None,
                 train=True, split=0.8):
        self.transitions = transitions
        if train:
            self.transitions = {k: v[:int(split * len(v))] for k, v in self.transitions.items()}
        else:
            self.transitions = {k: v[int(split * len(v)):] for k, v in self.transitions.items()}
        self.image = transitions['obs']
        self.forces = transitions['forces']
        self.features = transitions['features']
        self.transform_image = transform_image

    def __len__(self):
        return len(self.transitions['obs'])

    def __getitem__(self, idx):
        image = self.image[idx]
        if self.transform_image:
            image = self.transform_image(image)

        features = self.features[idx]
        force = self.forces[idx]
        force = np.expand_dims(force, axis=0)

        image = th.from_numpy(image).float()
        features = th.from_numpy(features).float()
        force = th.from_numpy(force).float()

        return image, features, force


class ForceNet(nn.Module):
    """ Cnn module

    :param input_channels: (int) Number of input channels
    :param features_dim: (int) Dimensionality of the features extracted by the CNN
    :param output_shape: (int) Dimensionality of the output
    """

    def __init__(
        self,
        input_channels: int = 1,
        features_dim: int = 768,
        output_shape: int = 1,
        activation_function: nn.Module = nn.ReLU(),
    ) -> None:
        super(ForceNet, self).__init__()
        self.example_input_array = th.Tensor(32, 1, 80, 80)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            activation_function,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            activation_function,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            activation_function,
            nn.Flatten(),
        )

        # Assume a dummy image input to compute the flattened size
        with th.no_grad():
            n_flatten = self.cnn(self.example_input_array).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(),)

        self.mlp = nn.Sequential(
            nn.Linear(features_dim, 256),
            activation_function,
            nn.Linear(256, output_shape),
            nn.ReLU(),
        )

    def forward(self, image: th.Tensor) -> th.Tensor:
        features = self.cnn(image)
        features = self.linear(features)
        output = self.mlp(features)
        return features, output


class ForcePrediction(pl.LightningModule):
    def __init__(self, input_channels: int = 1, features_dim: int = 768, output_shape: int = 1):
        super(ForcePrediction, self).__init__()
        self.force_net = ForceNet(input_channels, features_dim, output_shape)
        self.save_hyperparameters()
        self.l_features = nn.MSELoss()
        self.l_force = nn.MSELoss()
        self.name = 'feature_loss'

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features, output = self.force_net(observations)
        return features, output

    def predict_step(self, batch, batch_idx):
        image, features, force = batch
        _, force_pred = self(image)
        return force_pred

    def training_step(self, batch, batch_idx):
        image, features, force = batch
        features_pred, force_pred = self(image)
        l_features = self.l_features(features_pred, features)
        l_force = self.l_force(force_pred, force)
        loss = l_features + l_force
        loss = F.mse_loss(force_pred, force)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, features, force = batch
        _, force_pred = self(image)
        l_force = self.l_force(force_pred, force)
        self.log('val_loss', l_force)
        return l_force

    def test_step(self, batch, batch_idx):
        image, features, force = batch
        _, force_pred = self(image)
        l_force = self.l_force(force_pred, force)
        self.log('test_loss', l_force)
        return l_force

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=1e-4)
        return optimizer


class SimpleForcePrediction(ForcePrediction):
    def __init__(self, input_channels: int = 1, features_dim: int = 768, output_shape: int = 1):
        super(SimpleForcePrediction, self).__init__(input_channels, features_dim, output_shape)
        self.name = 'simple'

    def training_step(self, batch, batch_idx):
        image, features, force = batch
        _, force_pred = self(image)
        l_force = self.l_force(force_pred, force)
        self.log('train_loss', l_force)
        return l_force


class MeanForcePrediction(ForcePrediction):
    def __init__(self, input_channels: int = 1, features_dim: int = 768, output_shape: int = 1):
        super(MeanForcePrediction, self).__init__(input_channels, features_dim, output_shape)
        self.name = 'mean'

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features, output = self.force_net(observations)
        return features, th.Tensor([2.2816642032182624]).to(self.device)


class SELUActivation(ForcePrediction):
    def __init__(self, input_channels: int = 1, features_dim: int = 768, output_shape: int = 1):
        super(SELUActivation, self).__init__(input_channels, features_dim, output_shape)
        self.name = 'selu'
        self.force_net = ForceNet(input_channels, features_dim, output_shape, nn.SELU())

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=1e-4)
        # TODO: find a way to not hardcode this
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=1e-4,
                    epochs=30,  # this is hardcoded for now
                    steps_per_epoch=64048,  # this is hardcoded for now
                ),
            },
        }


class KLDivergence(ForcePrediction):
    def __init__(self, input_channels: int = 1, features_dim: int = 768, output_shape: int = 1):
        super(KLDivergence, self).__init__(input_channels, features_dim, output_shape)
        self.name = 'kl_divergence'
        self.l_features = nn.KLDivLoss(reduction='batchmean')


if __name__ == "__main__":
    # reproducibility
    th.manual_seed(0)
    seed = th.Generator().manual_seed(42)
    path = Path.cwd() / 'rl' / 'imitation' / 'trajectories' / 'bca_her_10000.npz'
    transitions = np.load(path, allow_pickle=True)
    transitions = clean_transitions(transitions, filter_keys=['forces', 'obs', 'features'])

    dataset = TransitionsDataset(transitions, train=True)
    test_dataset = TransitionsDataset(transitions, train=False)

    train_set_size = int(0.8 * len(dataset))
    val_set_size = len(dataset) - train_set_size
    train_set, val_set = th.utils.data.random_split(dataset, [train_set_size, val_set_size], generator=seed)

    num_cpu = os.cpu_count()
    train_loader = DataLoader(train_set, num_workers=num_cpu)
    val_loader = DataLoader(val_set, num_workers=num_cpu)
    test_loader = DataLoader(test_dataset, num_workers=num_cpu)

    test_trainer = pl.Trainer(fast_dev_run=True, max_epochs=10, logger=False)

    feature_dim = transitions['features'].shape[0]
    for model in [ForcePrediction, SimpleForcePrediction, MeanForcePrediction, SELUActivation]:
        model = model(input_channels=1, features_dim=feature_dim, output_shape=1)
        test_trainer.test(model, test_loader)
