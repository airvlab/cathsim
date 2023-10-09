import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def compute_regularization_loss(pred_points, distance=0.002):
    """The points should be at a distance of 0.002 from each other"""
    # Assuming pred_points is of shape [batch_size, N_pred, 3]

    # account for the possibility that the sequence is shorter than 2
    if pred_points.size(1) < 2:
        return torch.zeros(1).to(pred_points.device)
    diff = pred_points[:, 1:] - pred_points[:, :-1]
    dist = torch.sqrt((diff ** 2).sum(-1))

    return F.smooth_l1_loss(dist, torch.ones_like(dist) * distance)


def compute_loss(pred_points, target_points):
    # Assuming pred_points is of shape [batch_size, N_pred, 3] and
    # target_points is of shape [batch_size, N_target, 3]

    N_pred = pred_points.size(1)
    N_target = target_points.size(1)

    # Extend both sequences to have the max length of both
    N_max = max(N_pred, N_target)

    if N_pred < N_max:
        padding = torch.zeros(pred_points.size(0), N_max - N_pred, 3).to(pred_points.device)
        pred_points = torch.cat([pred_points, padding], dim=1)

    if N_target < N_max:
        padding = torch.zeros(target_points.size(0), N_max - N_target, 3).to(target_points.device)
        target_points = torch.cat([target_points, padding], dim=1)

    # Create a mask for valid points in the target
    mask = (target_points != 0).all(-1).float()

    # Huber loss
    pointwise_loss = F.smooth_l1_loss(pred_points, target_points, reduction='none')
    masked_loss = pointwise_loss * mask.unsqueeze(-1)

    reg_loss = compute_regularization_loss(pred_points)

    return 1000 * masked_loss.mean() + 0.1 * reg_loss


class ImageEncoder(nn.Module):
    def __init__(self, encoded_size=512):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, encoded_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, max_seq_length=86):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        positions = torch.arange(0, max_seq_length).unsqueeze(0)
        self.register_buffer('positions', positions)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Convert transformer output to predicted point
        self.fc_out = nn.Linear(d_model, 3)

    def forward(self, memory):
        batch_size = memory.size(0)
        memory = memory.unsqueeze(1)  # Add sequence length dimension
        pos = self.pos_encoder(self.positions[:, :memory.size(1)])  # Get positional encodings

        tgt = pos.repeat(batch_size, 1, 1)
        output = self.transformer_decoder(tgt, memory)
        return self.fc_out(output)


class ShapePredictionTransformer(nn.Module):
    def __init__(self, encoded_size=512, d_model=512, nhead=8, num_decoder_layers=6, max_seq_length=86):
        super(ShapePredictionTransformer, self).__init__()
        self.encoder = ImageEncoder(encoded_size)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, max_seq_length)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class ShapePredictionLightningTransformer(pl.LightningModule):
    def __init__(self, image_size=84, encoded_features=512, learning_rate=1e-4):
        super(ShapePredictionLightningTransformer, self).__init__()

        self.model = ShapePredictionTransformer(encoded_features)
        self.learning_rate = learning_rate
        self.example_input_array = torch.rand(1, 1, image_size, image_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted_sequences = self(inputs)

        loss = compute_loss(predicted_sequences, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)

        return {
            "optimizer": optimizer,
            "monitor": "train_loss",
        }


def test_model():
    x = torch.randn(1, 1, 80, 80)

    encoder = ImageEncoder()
    decoder = TransformerDecoder()

    features = encoder(x)
    print("\nEncoder output shape: ", features.shape, "\n")
    sequences = decoder(features)
    print("Decoder output shape: ", sequences.shape)

    random_sequence = torch.randn(1, 86, 3)
    loss = compute_loss(sequences, random_sequence)
    print("Loss: ", loss)

    reg_loss = compute_regularization_loss(sequences)
    print("Reg loss: ", reg_loss)


if __name__ == "__main__":
    test_model()
