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
    # train()
    # test()
    traj = Trajectory.load("test")
    print(traj)
    fig, ax = plt.subplots(1, figsize=(5.50 / 2, 5.50 / 2))
    image = plt.imread("phantom_480.png")
    traj.plot_path(
        ax,
        "plan_next_pos",
        plot_kwargs=dict(
            base_image=image,
            scatter_kwargs=dict(color="red", label="A* Plan"),
        ),
    )
    traj.plot_path(
        ax,
        "head_pos",
        plot_kwargs=dict(
            base_image=image,
            scatter_kwargs=dict(color="blue", label="Actual"),
        ),
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 0.95, 1, 0.2),
        # bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        # loc="outside upper center",
        # mode="expand",
        borderaxespad=0,
        frameon=False,
        ncol=2,
    )
    fig.tight_layout()
    plt.axis("off")
    # plt.tight_layout()
    plt.show()

    # plot_3D_to_2D(ax, *plan_next, color="r")
    # plt.show()
    # result = np.load("./scratch/astar_eval.npz", allow_pickle=True)
    # print(result.files[0]["results"])
