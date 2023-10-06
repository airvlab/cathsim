import cv2
import numpy as np
from pathlib import Path

from cathsim.rl.data import Trajectory
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

from shape_reconstruction import (
    get_equidistant_3d_points_from_images,
    read_segmented_image,
)


def pad_tensor(tensor, max_length, padding_value=-999):
    if tensor.size(0) < max_length:
        padding_size = max_length - tensor.size(0)
        padding = torch.full(
            (padding_size, 3), padding_value, dtype=torch.float, device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=0)
    return tensor


def collate_fn(batch):
    # Separate the images and the points
    images, points_list = zip(*batch)

    # Determine max length among all batches
    max_length = max([p.size(0) for p in points_list])

    # Pad each tensor in points to this max length
    padded_points = [pad_tensor(p, max_length) for p in points_list]

    # Stack everything up
    images = torch.stack(images)
    padded_points = torch.stack(padded_points)

    return images, padded_points


def clean_dataset(path: Path):
    for file in path.glob("*_actual.npy"):
        index = file.name.split("_")[0]
        top_path = path / f"{index}_top.jpg"
        side_path = path / f"{index}_side.jpg"
        actual_path = path / f"{index}_actual.npy"

        try:
            top_image = read_segmented_image(top_path)
            side_image = read_segmented_image(side_path)
            points = get_equidistant_3d_points_from_images(top_image, side_image)
            if len(points) < 10:
                print("Removing", index)
                top_path.unlink()
                side_path.unlink()
                actual_path.unlink()
        except IndexError:
            print("Removing", index)
            top_path.unlink()
            side_path.unlink()
            actual_path.unlink()


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


def pad_points(points: np.ndarray, max_length=84, padding_value=-999):
    padding_len = max_length - points.shape[0]
    padding = np.ones((padding_len, 3)) * padding_value
    padded_points = np.concatenate((points, padding), axis=0)

    return padded_points


def get_points(top_image, side_image):
    points = get_equidistant_3d_points_from_images(top_image, side_image)
    # points = pad_points(points)
    # points = torch.from_numpy(points)

    return points


class TransitionsDataset(data.Dataset):
    def __init__(self, path: Path, transform: callable = None):
        self.path = path
        self.indices = list(path.glob("*_top.jpg"))
        self.indices = [int(index.name.split("_")[0]) for index in self.indices]

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((84, 84)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        self.tops = []
        self.points = []
        for idx in range(len(self.indices)):
            top, point = self._get_item(idx)
            self.tops.append(top)
            self.points.append(point)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Fetch from pre-loaded data
        return self.tops[index], self.points[index]

    def get_paths(self, index):
        index = self.indices[index]

        top = self.path / f"{index}_top.jpg"
        side = self.path / f"{index}_side.jpg"
        actual = self.path / f"{index}_actual.npy"

        return top, side, actual

    def _get_item(self, index):
        top_path, side_path, _ = self.get_paths(index)

        top_image = read_segmented_image(top_path)
        side_image = read_segmented_image(side_path)

        points = get_points(top_image, side_image)
        points = torch.from_numpy(points)

        with Image.open(top_path) as top, Image.open(side_path) as side:
            if self.transform:
                top_image = self.transform(top)

        return top_image, points


def get_dataloader(path: Path, **kwargs) -> data.DataLoader:
    dataset = TransitionsDataset(Path.cwd() / "data")
    print("Loaded dataset with {} transitions".format(len(dataset)))
    return data.DataLoader(dataset, shuffle=True, **kwargs, collate_fn=collate_fn)


if __name__ == "__main__":
    clean_dataset(Path.cwd() / "data_3")
    exit()
    dataloader = get_dataloader(Path.cwd() / "data")
    for batch in dataloader:
        top, points = batch
        print(top.shape, points.shape)
        break
