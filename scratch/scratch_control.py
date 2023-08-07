from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from functools import reduce
from rl.data import Trajectory
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt


class TransitionsDataset(data.Dataset):
    def __init__(self, path: Path):
        trajectories = [Trajectory.load(t).flatten() for t in list(path.iterdir())]
        trajectories = [
            self.preprocess_trajectory(trajectory) for trajectory in trajectories
        ]
        trajectories = reduce(
            lambda acc, trajectory: acc + trajectory, trajectories, []
        )
        print(len(trajectories))
        self.trajectories = trajectories

    @staticmethod
    def preprocess_trajectory(
        trajectory: Trajectory,
    ) -> List[Tuple[List[Any], Any, List[Any]]]:
        act_key = "act"
        info_key = "info-head_pos"

        transitions = []
        for i in range(
            1, len(trajectory.data[act_key])
        ):  # start from 1 to skip first action
            head_pos_before = trajectory.data[info_key][i - 1].ravel().tolist()
            act = trajectory.data[act_key][i]
            head_pos_after = trajectory.data[info_key][i].ravel().tolist()
            transitions.append((head_pos_before, act, head_pos_after))

        return transitions

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        pos, act, next_pos = self.trajectories[index]

        pos = torch.from_numpy(np.array(pos)).float()
        act = torch.from_numpy(np.array(act)).float()
        next_pos = torch.from_numpy(np.array(next_pos)).float()

        return pos, act, next_pos


class MLP(pl.LightningModule):
    def __init__(
        self, input_dim: int = 6, output_dim: int = 2, hidden_dims: List[int] = [64, 64]
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(input_dim, hidden_dims[0])])
        self.layers.extend(
            [
                nn.Linear(hidden_dims[i - 1], hidden_dims[i])
                for i in range(1, len(hidden_dims))
            ]
        )
        self.layers.extend([nn.Linear(hidden_dims[-1], output_dim)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](
            x
        )  # no activation function as this is a regression problem

    def training_step(self, batch, batch_idx):
        pos, act, next_pos = batch
        x = torch.cat(
            [pos, next_pos], dim=-1
        )  # concatenate current and next positions as input
        y = act
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)  # MSE loss for regression problem
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def train():
    EPOCHS = 10
    BATCH_SIZE = 32

    path = Path.cwd() / Path("transitions/")
    dataset = TransitionsDataset(path)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLP()

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)

    wandb_logger = WandbLogger(
        project="control",
        name="simple",
        version="dev",
        job_type="train",
        save_dir=(Path.cwd() / Path("scratch/action_prediction/logs/")).as_posix(),
    )
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

    trainer = pl.Trainer(max_epochs=EPOCHS, logger=wandb_logger)
    trainer.fit(model, dataloader, ckpt_path="last")


def test():
    def evaluate_policy(model, env, plan, n_episodes: int = 1) -> dict:
        plan = [np.array(pos) for pos in plan.data["head_pos"]]
        observation, new_observation = plan[:-1], plan[1:]
        traj = Trajectory()
        for episode in tqdm(range(n_episodes)):
            env.reset()
            done = False
            head_pos = env.head_pos.copy()
            for obs, new_obs in zip(observation, new_observation):
                obs, new_obs = (
                    torch.from_numpy(obs).float(),
                    torch.from_numpy(new_obs).float(),
                )
                action = model(torch.cat([obs, new_obs]))
                action = action.detach().numpy()
                _, reward, done, info = env.step(action)
                traj.add_transition(
                    plan_pos=obs,
                    act=action,
                    plan_next_pos=new_obs,
                    head_pos=head_pos,
                    force=info["forces"],
                )
                head_pos = info["head_pos"]
            traj.save("test")

    from cathsim.cathsim.env_utils import make_gym_env
    from rl.utils import get_config

    ckpt_path = Path("lightning_logs/version_1/checkpoints/epoch=9-step=110080.ckpt")
    model = MLP.load_from_checkpoint(ckpt_path, map_location="cpu")

    config = get_config("internal")
    env = make_gym_env(config)

    plan = Trajectory.load(Path("./astar_path"))
    print(plan)
    evaluate_policy(model, env, plan)
    # np.savez("./scratch/astar_eval", results=evaluation_data)


if __name__ == "__main__":
    from cathsim.cathsim.common import point2pixel

    def plot_3D_to_2D(
        ax,
        data,
        add_line: bool = True,
        matrix: int = 80,
        line_kwargs: dict = {},
        scatter_kwargs: dict = {},
    ) -> plt.Axes:
        camera_matrix = {
            480: np.array(
                [
                    [-5.79411255e02, 0.00000000e00, 2.39500000e02, -5.33073376e01],
                    [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.08351407e02],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, -1.50000000e-01],
                ]
            ),
            80: np.array(
                [
                    [-96.56854249, 0.0, 39.5, -8.82205627],
                    [0.0, 96.56854249, 39.5, -17.99606781],
                    [0.0, 0.0, 1.0, -0.15],
                ]
            ),
        }
        data = np.apply_along_axis(
            point2pixel, 1, data, camera_matrix=camera_matrix[matrix]
        )
        data = [point for point in data if np.all((0 <= point) & (point <= matrix))]
        data = np.array(data)  # Convert back to numpy array
        data[:, 1] = matrix - data[:, 1]
        ax.scatter(data[:, 0], data[:, 1], **scatter_kwargs)
        if add_line:
            ax.plot(data[:, 0], data[:, 1], **line_kwargs)
        return ax

    # train()
    # test()
    traj = Trajectory.load("test")
    print(traj)
    plan_next = list(traj["plan_next"].values())
    head_pos = list(traj["head_pos"].values())
    fig, ax = plt.subplots(1)
    image = plt.imread("phantom_480.png")
    image = np.flipud(image)
    plt.imshow(image)
    image_size = image.shape[0]
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    scatter_kwargs = {"color": "blue"}
    line_kwargs = {"color": "black"}
    plot_3D_to_2D(
        ax,
        *plan_next,
        matrix=image_size,
        scatter_kwargs={"color": "red"},
        line_kwargs=line_kwargs
    )
    plot_3D_to_2D(
        ax,
        *head_pos,
        matrix=image_size,
        scatter_kwargs=scatter_kwargs,
        line_kwargs=line_kwargs
    )
    plt.show()

    # plot_3D_to_2D(ax, *plan_next, color="r")
    # plt.show()
    # result = np.load("./scratch/astar_eval.npz", allow_pickle=True)
    # print(result.files[0]["results"])
