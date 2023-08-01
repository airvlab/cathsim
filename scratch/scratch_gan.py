from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils import data

from rl.data import TrajectoriesDataset, plot_path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

################################# CONSTANTS ###################################


class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=300 * 3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z, start_point, end_point):
        x = torch.cat([start_point, end_point], dim=1)
        trajectory = self.model(x)
        return trajectory.view(trajectory.size(0), 300, 3)


class Discriminator(nn.Module):
    def __init__(self, input_dim=300 * 3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, trajectory):
        trajectory = trajectory.view(
            trajectory.size(0), -1
        )  # reshape to (batch_size, 300*3)
        validity = self.model(trajectory)
        return validity


class LightningGan(pl.LightningModule):
    def __init__(self, generator, discriminator, learning_rate=0.001):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.bce_loss = nn.BCELoss()
        self.automatic_optimization = False
        self.noise_dim = 0
        # self.save_hyperparameters()

    def forward(self, noise, start, end):
        return self.generator(noise, start, end)

    def configure_optimizers(self):
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        return self.optimizer_G, self.optimizer_D

    @staticmethod
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def training_step(self, batch, batch_idx):
        (start_points, end_points), real_trajectories = batch
        batch_size = real_trajectories.size(0)

        # Adversarial ground truths
        real = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Configure input
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)

        # -----------------
        #  Train Generator
        # -----------------
        self.optimizers()[0].zero_grad()

        # Generate a batch of trajectories
        generated_trajectories = self.generator(noise, start_points, end_points)

        # Loss measures generator's ability to fool the discriminator
        g_loss = self.bce_loss(self.discriminator(generated_trajectories), real)

        self.manual_backward(g_loss)
        self.optimizers()[0].step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizers()[1].zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.bce_loss(self.discriminator(real_trajectories), real)
        fake_loss = self.bce_loss(
            self.discriminator(generated_trajectories.detach()), fake
        )
        d_loss = (real_loss + fake_loss) / 2

        self.manual_backward(d_loss)
        self.optimizers()[1].step()

        # Logging to TensorBoard by default
        self.log(
            "generator_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "discriminator_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


def cmd_train_model(args):
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument("--path", type=str, default="transitions/")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    path = Path.cwd() / Path(args.path)
    dataset = TrajectoriesDataset(path, lazy_load=False)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size)

    generator = Generator(noise_dim=0)
    discriminator = Discriminator()

    model = LightningGan(generator, discriminator)

    wandb_logger = WandbLogger(
        name="simple",
        version="dev",
        job_type="train",
        save_dir=(Path.cwd() / Path("scratch/path_generation/logs/")).as_posix(),
    )
    wandb_logger.experiment.config["batch_size"] = args.batch_size

    trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger)
    trainer.fit(model, dataloader, ckpt_path="last")


if __name__ == "__main__":
    ckpt_path = Path(
        "scratch/path_generation/logs/lightning_logs/dev/checkpoints/epoch=799-step=1774400.ckpt"
    )
    generator = Generator(noise_dim=0)
    discriminator = Discriminator()

    model = LightningGan.load_from_checkpoint(
        ckpt_path, map_location="cpu", generator=generator, discriminator=discriminator
    )

    path = Path.cwd() / Path("transitions/")
    dataset = TrajectoriesDataset(path, lazy_load=False)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

    batch = next(iter(dataloader))
    noise = torch.randn(1, 0, device=model.device)

    fig, ax = plt.subplots(1, 1)
    # base_image = plt.imread("phantom_480.png")
    # plt.imshow(base_image)
    # image_height = base_image.shape[0]
    (start, end), trajectory = batch
    pred_path = model(noise, start, end)[2]
    trajectory = trajectory.detach().numpy()[0]
    print(trajectory.shape, trajectory.min(), trajectory.max())
    plot_path(ax, trajectory, scatter_kwargs={"c": "blue"})
    # trajectory[:, 1] = image_height - (trajectory[:, 1] * image_height)
    # pred_path[:, 1] = image_height - (pred_path[:, 1] * image_height)
    # plot_path(ax, pred_path.detach().numpy(), scatter_kwargs={"c": "red"})
    plot_path(ax, trajectory, scatter_kwargs={"c": "blue"})
    ax.set_ybound(0, 480)
    ax.set_xbound(0, 480)

    # flip the plot

    plt.show()
