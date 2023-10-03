import cv2
import numpy as np
from pathlib import Path

from cathsim.rl.data import Trajectory
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


image_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Lambda(lambda image: image.permute(2, 0, 1)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def preprocess_trajectory(traj: Trajectory):
    top = traj.data["top"]
    side = traj.data["side"]
    actual = traj.data["geom_pos"]

    transitions = []
    for i in range(len(traj)):
        if np.all(actual[i] == 0):
            continue
        transitions.append((top[i], side[i], actual[i]))

    return transitions


class TransitionsDataset(data.Dataset):
    def __init__(self, path: Path, transform: callable = None):
        self.path = path
        # Set default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        count = sum(1 for _ in self.path.glob("*_top.jpg"))
        return count

    def __getitem__(self, index):
        top, side, actual = self._get_item(index)

        # If self.transform is not None, it will be applied in _get_item method
        return top, side, actual

    def get_paths(self, index):
        top = self.path / f"{index}_top.jpg"
        side = self.path / f"{index}_side.jpg"
        actual = self.path / f"{index}_actual.npy"

        return top, side, actual

    # TODO: Will have to get the 3D points from the images
    def _get_item(self, index):
        top_path, side_path, actual_path = self.get_paths(index)

        actual = np.load(actual_path.as_posix())
        actual = torch.from_numpy(actual).float()

        with Image.open(top_path) as top, Image.open(side_path) as side:
            if self.transform:
                top = self.transform(top)
                side = self.transform(side)

        return top, side, actual


if __name__ == "__main__":
    dataset = TransitionsDataset(Path.cwd() / "data")
    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataloader:
        top, side, actual = batch
        print(top.shape, side.shape, actual.shape)
        break
