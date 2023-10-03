from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch.utils import data

from metrics import (
    mean_squared_error,
)

from models import (
    ShapePredictionLightning,
)

from dataset import TransitionsDataset


def train():
    EPOCHS = 30
    BATCH_SIZE = 16

    path = Path.cwd() / Path("transitions_bodies-sample/")
    dataset = TransitionsDataset(path)
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
    ckpt_path = Path(
        "scratch/shape_prediction/shape/dev/checkpoints/epoch=29-step=74910.ckpt"
    )
    model = ShapePredictionLightning.load_from_checkpoint(ckpt_path, map_location="cpu")

    BATCH_SIZE = 32

    dataset = TransitionsDataset(Path.cwd() / "data")
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch in dataloader:
        obs, guidewire_geom_pos = batch
        guidewire_geom_pos_pred = model(obs)
        for indice in range(10):
            pos = guidewire_geom_pos[indice].detach().numpy()
            pos_pred = guidewire_geom_pos_pred[indice].detach().numpy()


if __name__ == "__main__":
    train()
