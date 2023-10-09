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


class EncoderCNN(nn.Module):
    def __init__(self, image_size: int = 480, out_features: int = 512):
        super(EncoderCNN, self).__init__()

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
            nn.Linear(128 * 10 * 10, out_features),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LengthPredictor(nn.Module):
    def __init__(self, in_features: int = 512):
        super(LengthPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Ensure the output is between 0 and 1
        )

    def forward(self, x):
        # Output will be in range [0, 1], scale it to [1, 85]
        seq_length_fraction = self.predictor(x)
        seq_lengths = (seq_length_fraction * 84 + 1).squeeze().long()

        return seq_lengths.view(-1)  # Ensures it's a 1D tensor


class DecoderLSTM(nn.Module):
    def __init__(self, in_features, hidden_size=512, num_layers=2, max_seq_length=84):
        super(DecoderLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_features,  # Change this to match the encoder output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.point_classifier = nn.Linear(hidden_size, 3)
        self.max_seq_length = max_seq_length

    def forward(self, features, seq_lengths):
        batch_size = features.size(0)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(features.device)

        # Check if LSTM's expected input size matches the feature size
        lstm_expected_input_size = self.lstm._flat_weights[0].size(0) // 4
        if features.size(1) != lstm_expected_input_size:
            raise ValueError(f"Feature size ({features.size(1)}) does not match LSTM's expected input size ({lstm_expected_input_size}).")

        # Start token (initialized with zeros)
        lstm_input = torch.zeros(batch_size, 1, features.size(1)).to(features.device)

        # List to store LSTM outputs
        lstm_outs = []

        # Loop through LSTM for maximum sequence length
        for step in range(self.max_seq_length):
            self.lstm.flatten_parameters()
            lstm_out, (h0, c0) = self.lstm(lstm_input, (h0, c0))  # Use and update hidden and cell states
            point = self.point_classifier(lstm_out)
            lstm_outs.append(point)

            # Set next input to current output
            lstm_input = point

        # Stack the LSTM outputs
        sequences = torch.cat(lstm_outs, dim=1)

        # Mask sequences based on the predicted sequence lengths
        mask = self.generate_mask(seq_lengths, self.max_seq_length).unsqueeze(-1).to(features.device)
        sequences = sequences * mask.float()

        return sequences

    @staticmethod
    def generate_mask(seq_lengths, max_length):
        """ Generate a mask for the sequences """
        batch_size = seq_lengths.size(0)
        mask = torch.arange(max_length).unsqueeze(0).repeat(batch_size, 1)
        mask = mask < seq_lengths.unsqueeze(1)
        return mask


class ShapePredictionLightningLSTM(pl.LightningModule):
    def __init__(self, image_size=84, encoded_features=512, learning_rate=1e-4):
        super(ShapePredictionLightningLSTM, self).__init__()

        # Initialize the components
        self.encoder = EncoderCNN(image_size=image_size, out_features=encoded_features)
        self.length_predictor = LengthPredictor(encoded_features)
        self.decoder = DecoderLSTM(encoded_features)
        self.learning_rate = learning_rate
        self.example_input_array = torch.rand(1, 1, image_size, image_size)

    def forward(self, x):
        # Encode the image
        encoded_features = self.encoder(x)
        print("Encoded features shape: ", encoded_features.shape)
        # Predict the sequence length
        seq_lengths = self.length_predictor(encoded_features)
        print("Sequence lengths: ", seq_lengths)
        # Decode the features into sequences
        sequences = self.decoder(encoded_features, seq_lengths)
        print("Sequences shape: ", sequences.shape)

        return sequences, seq_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted_sequences, predicted_lengths = self(inputs)

        # Compute the custom loss you provided earlier
        huber_loss, reg_loss, loss = self.custom_loss(predicted_sequences, targets)

        self.log("train_loss", loss)
        self.log("huber", huber_loss)
        self.log("reg", reg_loss)
        return loss

    @staticmethod
    def custom_loss(predicted_sequences, predicted_lengths, target, alpha=1_000, beta=1):
        exit()
        huber_losses = torch.zeros(predicted_sequences.size(0)).to(predicted_sequences.device)
        reg_losses = torch.zeros(predicted_sequences.size(0)).to(predicted_sequences.device)

        # Looping over each item in the batch
        for i in range(predicted_sequences.size(0)):
            # Extracting sequences based on predicted length
            pred_sequence = predicted_sequences[i, :predicted_lengths[i]]
            target_sequence = target[i, :predicted_lengths[i]]

            # Print shapes for debugging
            print(f"Item {i} - Pred sequence shape: {pred_sequence.shape}")
            print(f"Item {i} - Target sequence shape: {target_sequence.shape}")

            # Compute the Huber loss for this sequence
            huber_losses[i] = F.smooth_l1_loss(pred_sequence, target_sequence, reduction='mean')

            # Regularization loss for this sequence
            if pred_sequence.size(0) > 1:
                diff = pred_sequence[1:] - pred_sequence[:-1]
                distances = torch.norm(diff, dim=-1)
                deviation = torch.abs(distances - 0.002)
                reg_losses[i] = torch.mean(deviation)

        # Averaging the losses over the batch
        avg_huber_loss = torch.mean(huber_losses)
        avg_reg_loss = torch.mean(reg_losses)

        # Combined loss
        combined_loss = alpha * avg_huber_loss + beta * avg_reg_loss
        return avg_huber_loss, avg_reg_loss, combined_loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
            "monitor": "train_loss",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


def get_model():
    pass


def test_model():
    image_size = 84
    x = torch.rand(1, 1, 84, 84)

    encoder = EncoderCNN(image_size=84, out_features=512)

    x_encoded = encoder(x)
    # print(x_encoded.shape)


if __name__ == "__main__":
    model = ShapePredictionLightningLSTM()
    x = torch.rand(2, 1, 84, 84)
    y, l = model(x)
    print(y.shape)
    print(l.shape)
    print(l)
    test_model()
