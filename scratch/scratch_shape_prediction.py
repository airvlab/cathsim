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
from rl.data import Trajectory
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


def train():
    EPOCHS = 30
    BATCH_SIZE = 16

    path = Path.cwd() / Path("transitions_bodies-sample/")
    dataset = TransitionsDataset(path, transform=image_transforms)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ShapePredictionLightning()

    wandb_logger = WandbLogger(
        project="shape",
        name="nadam",
        version="dev",
        job_type="train",
        save_dir=(Path.cwd() / Path("scratch/shape_prediction/")).as_posix(),
    )
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

    trainer = pl.Trainer(max_epochs=EPOCHS, logger=wandb_logger)
    trainer.fit(model, dataloader, ckpt_path="last")


def test():
    def plot_mesh(ax: plt.Axes, mesh: Union[trimesh.Trimesh, Path] = None):
        if mesh is None:
            mesh_path = Path.cwd() / Path(
                "cathsim/assets/meshes/phantom3/visual.stl"
            )
            mesh = trimesh.load_mesh(mesh_path)
        vertices = mesh.vertices
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            triangles=mesh.faces,
            Z=vertices[:, 2],
            color=[0.1 for i in range(4)],
        )
        ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
        ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
        ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])

    def plot3D(
        points: list, ax: plt.Axes, c: str = "b", mesh: trimesh.Trimesh = None, **kwargs
    ):
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, s=1, **kwargs)

    ckpt_path = Path(
        "scratch/shape_prediction/shape/dev/checkpoints/epoch=29-step=74910.ckpt"
    )
    model = ShapePredictionLightning.load_from_checkpoint(ckpt_path, map_location="cpu")

    BATCH_SIZE = 32

    path = Path.cwd() / Path("transitions_bodies-sample/")
    dataset = TransitionsDataset(path, transform=image_transforms)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    mesh = trimesh.load_mesh(Path("cathsim/assets/meshes/phantom3/visual.stl"))
    for batch in dataloader:
        obs, guidewire_geom_pos = batch
        guidewire_geom_pos_pred = model(obs)
        for indice in range(10):
            pos = guidewire_geom_pos[indice].detach().numpy()
            pos_pred = guidewire_geom_pos_pred[indice].detach().numpy()
            PAGE_SIZE = 5.50
            fig = plt.figure(figsize=(PAGE_SIZE * 0.6, PAGE_SIZE * 0.6))
            ax = fig.add_subplot(111, projection="3d")
            plot_mesh(ax, mesh=mesh)
            plot3D(pos, ax, c="r", label="Real")
            plot3D(pos_pred, ax, label="Predicted")

            plt.tight_layout(h_pad=10)
            plt.legend(
                # bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="upper center",
                ncols=2,
                # mode="expand",
                borderaxespad=0.0,
                frameon=False,
            )
            plt.setp(ax.get_zticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.savefig("shape_prediction.png", dpi=300)
            plt.show()


if __name__ == "__main__":
    # train()
    test()
