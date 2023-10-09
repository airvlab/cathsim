from visualization import visualize_3d

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from models import (
    ShapePredictionLightning,
    ShapePredictionLightningLSTM,
)

from lightning_models import ShapePredictionLightningTransformer

from dataset import get_dataloader


def get_logger(BATCH_SIZE):
    wandb_logger = WandbLogger(
        project="shape",
        name="nadam",
        version="dev",
        job_type="train",
        save_dir=(Path.cwd() / Path("scratch/shape_prediction/")).as_posix(),
    )
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE

    return wandb_logger


def debug():
    BATCH_SIZE = 32

    dataloader = get_dataloader(Path.cwd() / Path("data_2"), batch_size=2)
    model = ShapePredictionLightningTransformer()
    # model = ShapePredictionLightningLSTM(84, learning_rate=1e-4)

    trainer = pl.Trainer(fast_dev_run=2)
    trainer.fit(model, dataloader)


def train():
    EPOCHS = 100
    BATCH_SIZE = 16

    path = Path.cwd() / Path("data_3")
    print("Training on: ", path)

    dataloader = get_dataloader(path, batch_size=BATCH_SIZE)

    model = ShapePredictionLightning(learning_rate=1e-4)

    # wandb_logger = get_logger(BATCH_SIZE)

    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model, dataloader, ckpt_path="last")


def get_model():
    version = 7
    ckpt_path = Path.cwd() / f"lightning_logs/version_{version}/checkpoints"
    ckpt_path = ckpt_path.glob("*.ckpt")
    ckpt_path = list(ckpt_path)[0]
    model = ShapePredictionLightning.load_from_checkpoint(ckpt_path, map_location="cpu")
    return model


def test():
    version = 7
    ckpt_path = Path.cwd() / f"lightning_logs/version_{version}/checkpoints"
    ckpt_path = ckpt_path.glob("*.ckpt")
    ckpt_path = list(ckpt_path)[0]
    model = ShapePredictionLightning.load_from_checkpoint(ckpt_path, map_location="cpu")

    BATCH_SIZE = 16

    dataloader = get_dataloader(Path.cwd() / Path("data_2"), batch_size=BATCH_SIZE)

    for batch in dataloader:
        top, points = batch
        points_pred = model(top)
        for indice in range(10):
            point = points[indice].detach().numpy()
            point_pred = points_pred[indice].detach().numpy()
            print(point_pred.shape, point.shape)

            point_pred = point_pred[: point.shape[0]]
            visualize_3d([point, point_pred], ["actual", "pred"])
            print("Predicted shape: ", point_pred.shape)
            print("Actual shape: ", point.shape)

        exit()


if __name__ == "__main__":
    # debug()
    # train()
    test()
