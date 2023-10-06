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


class ShapePredictionCNN(nn.Module):
    def __init__(self):
        super(ShapePredictionCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
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
        self.example_input_array = torch.rand(1, 1, 84, 84)

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
    def custom_loss(
        predicted, target, max_length=84, alpha=1_000, beta=1, padding_value=-999
    ):
        # Pad the target points
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

        # Combined loss
        combined_loss = alpha * huber_loss + beta * reg_loss
        return alpha * huber_loss, beta * reg_loss, combined_loss

    # @staticmethod
    # def custom_loss(predicted, target, alpha=1_000, beta=1):
    #     huber_loss = F.smooth_l1_loss(predicted, target)
    #
    #     diff = predicted[:, 1:, :] - predicted[:, :-1, :]
    #     distances = torch.norm(diff, dim=-1)
    #     deviation = torch.abs(distances - 0.002)
    #     reg_loss = torch.mean(deviation)
    #
    #     combined_loss = alpha * huber_loss + beta * reg_loss
    #
    #     return alpha * huber_loss, beta * reg_loss, combined_loss

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


def get_model():
    pass


if __name__ == "__main__":
    pass
