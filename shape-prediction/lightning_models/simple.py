import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F


def pad_tensor(tensor, max_length, padding_value=-999):
    """Pad tensor to max_length with padding_value."""
    padding_len = max_length - tensor.size(0)
    padding_dims = (0, 0, 0, padding_len)  # (left, right, top, bottom) for 2D padding
    padded_tensor = torch.nn.functional.pad(
        tensor, padding_dims, "constant", padding_value
    )
    return padded_tensor


def loss_fn(predicted, target, max_length=84, alpha=1_000, beta=1, padding_value=-999):
    padded_target = [pad_tensor(t, max_length, padding_value) for t in target]
    padded_target = torch.stack(padded_target).to(predicted.device)

    # Mask for valid points
    valid_mask = (padded_target != padding_value).all(dim=-1)
    expanded_mask = valid_mask.unsqueeze(-1).expand_as(predicted)

    # Mask the predictions and the target
    masked_pred = predicted[expanded_mask].view(-1, 3)
    masked_target = padded_target[expanded_mask].view(-1, 3)

    # Compute the huber loss
    huber_loss = F.smooth_l1_loss(masked_pred, masked_target)

    # Regularization
    diff = masked_pred[1:] - masked_pred[:-1]
    distances = torch.norm(diff, dim=-1)
    deviation = torch.abs(distances - 0.002)
    reg_loss = torch.mean(deviation)

    combined_loss = alpha * huber_loss + beta * reg_loss
    return alpha * huber_loss, beta * reg_loss, combined_loss


class ShapePredictionCNN(nn.Module):
    def __init__(self, img_size=84):
        super(ShapePredictionCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            sample = torch.randn((1, 1, img_size, img_size))
            output = self.features(sample)
            flattened_size = output.view(output.size(0), -1).size(1)

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 252),
            nn.ReLU(),
            nn.Linear(252, 84 * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.view(x.size(0), 84, 3)
        return x


class ShapePredictionLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, img_size=84):
        super(ShapePredictionLightning, self).__init__()
        self.model = ShapePredictionCNN(img_size=img_size)
        self.learning_rate = learning_rate
        self.example_input_array = torch.rand(1, 1, img_size, img_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        huber_loss, reg_loss, loss = loss_fn(outputs, targets)
        self.log("train_loss", loss)
        self.log("huber", huber_loss)
        self.log("reg", reg_loss)

        return loss

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


def test_model():
    img_size = 480
    x = torch.rand(1, 1, img_size, img_size)

    model = ShapePredictionLightning(img_size=img_size)
    model.eval()
    model.freeze()

    y = model(x)
    print(y.shape)

    loss = loss_fn(y, [torch.rand(84, 3)])
    print(loss)


if __name__ == "__main__":
    test_model()
