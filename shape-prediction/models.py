from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from functools import reduce
from cathsim.rl.data import Trajectory
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import trimesh

image_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda image: image.permute(2, 0, 1)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class TransitionsDataset(data.Dataset):
    def __init__(self, path: Path, transform: callable = None):
        trajectories = [
            Trajectory.load(t).to_array().flatten() for t in list(path.iterdir())
        ]
        trajectories = [
            self.preprocess_trajectory(trajectory) for trajectory in trajectories
        ]
        trajectories = reduce(
            lambda acc, trajectory: acc + trajectory, trajectories, []
        )
        self.transform = transform
        self.trajectories = trajectories

    @staticmethod
    def preprocess_trajectory(
        trajectory: Trajectory,
    ) -> List[Tuple[List[Any], List[Any]]]:
        obs_key = "next_obs-pixels"
        info_key = "info-guidewire_geom_pos"

        transitions = []
        for i in range(len(trajectory.data[obs_key])):
            obs = trajectory.data[obs_key][i]
            guidewire_geom_pos = trajectory.data[info_key][i]
            transitions.append((obs, guidewire_geom_pos))

        return transitions

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        image, guidewire_geom_pos = self.trajectories[index]

        image = torch.from_numpy(np.array(image)).float()
        if self.transform:
            image = self.transform(image)
        guidewire_geom_pos = torch.from_numpy(np.array(guidewire_geom_pos)).float()

        return image, guidewire_geom_pos


class ShapePredictionCNN(nn.Module):
    def __init__(self):
        super(ShapePredictionCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # Adjusted in_channels to 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 2
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 3
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 252),
            nn.ReLU(),
            nn.Linear(252, 84 * 3),
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # Reshape the output tensor
        x = x.view(x.size(0), 84, 3)
        return x


class ShapePredictionLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(ShapePredictionLightning, self).__init__()
        self.model = ShapePredictionCNN()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        huber_loss, reg_loss, loss = self.custom_loss(outputs, targets)
        self.log("train_loss", loss)
        self.log("huber", huber_loss)
        self.log("reg", reg_loss)

        return loss

    @staticmethod
    def custom_loss(predicted, target, alpha=1_000, beta=1):
        huber_loss = F.smooth_l1_loss(predicted, target)

        diff = predicted[:, 1:, :] - predicted[:, :-1, :]
        distances = torch.norm(diff, dim=-1)
        deviation = torch.abs(distances - 0.002)
        reg_loss = torch.mean(deviation)

        combined_loss = alpha * huber_loss + beta * reg_loss

        return alpha * huber_loss, beta * reg_loss, combined_loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.model.parameters(), lr=self.learning_rate)

        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
            "monitor": "train_loss",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
