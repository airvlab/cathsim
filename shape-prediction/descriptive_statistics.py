from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image

from cathsim.dm.visualization import point2pixel


from shape_reconstruction import (
    read_segmented_image,
    get_equidistant_3d_points_from_images,
    P_TOP,
    P_SIDE,
)

from visualization import (
    visualize_images,
    visualize_projections,
    make_3d_figure,
    make_2d_figure,
    make_3d_heatmap_figure,
    make_error_figure,
)


transform = transforms.Compose(
    [
        transforms.Resize((84, 84)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)


def is_within_image(point, image_shape):
    return np.all(point >= 0) and point[0] < image_shape and point[1] < image_shape


def main_2():
    from metrics import get_stats
    from train import get_model

    path = Path.cwd() / "data_2"
    print("Analyzing data from", path)

    def get_valid_points(points):
        reprojected_top = point2pixel(points, P_TOP)
        reprojected_side = point2pixel(points, P_SIDE)

        valid_top = np.array([is_within_image(pt, 480) for pt in reprojected_top])
        valid_side = np.array([is_within_image(pt, 480) for pt in reprojected_side])

        valid_points = np.logical_and(valid_top, valid_side)
        return points[valid_points]

    def load_data():
        indices = [i.name.split("_")[0] for i in path.glob("*_actual.npy")][:10]
        top, side, actual, pred = [], [], [], []
        for i in indices:
            top.append(plt.imread(path / f"{i}_top.jpg"))
            side.append(plt.imread(path / f"{i}_side.jpg"))

            pred_points = np.load(path / f"{i}_pred_points.npy")
            actual_points = get_valid_points(np.load(path / f"{i}_actual.npy"))
            actual_points = np.flip(actual_points, axis=0)

            min_len = min(len(pred_points), len(actual_points))
            pred_points = pred_points[:min_len]
            actual_points = actual_points[:min_len]

            actual.append(actual_points)
            pred.append(pred_points)

        assert len(top) == len(side) == len(actual) == len(pred)
        return top, side, actual, pred

    model = get_model()
    model.eval()
    model.freeze()

    top, side, actual, triangulated = load_data()
    torch_top = torch.zeros((len(top), 1, 84, 84))
    for i in range(len(top)):
        image = Image.fromarray(top[i])
        torch_top[i] = transform(image)

    pred = model(torch_top)
    pred_points = []

    for i in range(len(pred)):
        pred_i = pred[i].detach().cpu().numpy()
        min_len = min(len(pred[i]), len(triangulated[i]))
        pred_i = pred_i[:min_len]
        pred_points.append(pred_i)

    for i in range(len(triangulated)):
        reprojected_top_pred = point2pixel(triangulated[i], P_TOP)
        reprojected_top_actual = point2pixel(actual[i], P_TOP)
        reprojected_top_network = point2pixel(pred_points[i], P_TOP)
        plt.imshow(top[i])
        plt.scatter(
            reprojected_top_pred[:, 0],
            reprojected_top_pred[:, 1],
            label="Reconstructed",
        )
        plt.scatter(
            reprojected_top_actual[:, 0],
            reprojected_top_actual[:, 1],
            label="Actual",
        )

        plt.scatter(
            reprojected_top_network[:, 0],
            reprojected_top_network[:, 1],
            label="Network",
        )

        plt.axis("off")
        plt.show()

    stats = get_stats(triangulated, actual)
    print("Triangulated vs Actual")
    __import__("pprint").pprint(stats)
    print("\nNetwork vs Actual")
    stats = get_stats(pred_points, actual)
    __import__("pprint").pprint(stats)

    pass


if __name__ == "__main__":
    main_2()
    exit()
    main()
    exit()
    indice = 10
    top, side, actual = load_data()
    print(top.shape, side.shape, len(actual))
    top, side, actual = filter_data(top, side, actual, threshold=30)
    print(top.shape, side.shape, len(actual))
    top, side, actual = top[indice], side[indice], actual[indice]
    points = get_equidistant_3d_points_from_images(top, side)
    visualize_projections([top] * 2, [side] * 2, [points, actual], n=2)
