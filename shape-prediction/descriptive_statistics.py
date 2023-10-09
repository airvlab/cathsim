from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from cathsim.dm.visualization import point2pixel

from metrics import *

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

from train import get_model


def is_within_image(point, image_shape):
    return np.all(point >= 0) and point[0] < image_shape and point[1] < image_shape


def filter_data(indices, actual, threshold=10):
    indices_filtered, filtered_actual = [], []

    for indice, points in zip(indices, actual):
        reprojected_top = point2pixel(points, P_TOP)
        reprojected_side = point2pixel(points, P_SIDE)

        valid_top = np.array([is_within_image(pt, 480) for pt in reprojected_top])
        valid_side = np.array([is_within_image(pt, 480) for pt in reprojected_side])

        valid_points = np.logical_and(valid_top, valid_side)

        if np.sum(valid_top) > threshold and np.sum(valid_side) > threshold:
            filtered_actual.append(points[valid_points])
            indices_filtered.append(indice)

    return indices_filtered, filtered_actual


def get_predicted_points(indices, actual, path):
    indices_filtered, filtered_actual, predicted = [], [], []
    for indice, points in zip(indices, actual):
        try:
            top_image = read_segmented_image(path / f"{indice}_top.jpg")
            side_image = read_segmented_image(path / f"{indice}_side.jpg")
            pred_points = get_equidistant_3d_points_from_images(top_image, side_image)
            print(
                indice,
                top_image.shape,
                side_image.shape,
                points.shape,
                pred_points.shape,
                end="\r",
            )
            indices_filtered.append(indice)
            filtered_actual.append(points)
            predicted.append(pred_points)
        except IndexError:
            continue

    return (
        indices_filtered,
        predicted,
        filtered_actual,
    )


def load_data(path):

    indices = [i.name.split("_")[0] for i in path.glob("*_actual.npy")][:100]

    actual = []
    for i in indices:
        shape = np.load(path / f"{i}_actual.npy")
        reversed_shape = np.flip(shape, axis=0)
        assert shape.shape == reversed_shape.shape
        actual.append(reversed_shape)

    return indices, np.array(actual)


def load_images(indices, path):
    top_real, side_real, top_mask, side_mask = [], [], [], []
    for i in indices:
        # top_real.append(plt.imread(path / f"{i}_top_real.jpg"))
        # side_real.append(plt.imread(path / f"{i}_side_real.jpg"))
        top_mask.append(read_segmented_image(path / f"{i}_top.jpg"))
        side_mask.append(read_segmented_image(path / f"{i}_side.jpg"))
    return top_mask, side_mask


def main():
    from metrics import get_stats
    from optimization import get_curve_3D, sample_points

    path = Path.cwd() / "data_3"

    indices, actual = load_data(path)
    indices, actual = filter_data(indices, actual, threshold=30)
    indices, pred, actual = get_predicted_points(indices, actual, path)

    for i in range(len(pred)):
        curve = get_curve_3D(pred[i])
        pred[i] = sample_points(curve, 0.002)
        min_len = min(len(pred[i]), len(actual[i]))

        pred[i] = pred[i][:min_len]
        actual[i] = actual[i][:min_len]

    stats = get_stats(pred, actual)
    __import__("pprint").pprint(stats)

    top_mask, side_mask = load_images(indices, path)
    # visualize_projections(top, side, actual, n=3)
    # visualize_projections(top, side, pred, n=3)
    i = 33
    predi, actuali = pred[i], actual[i]
    # make_3d_figure([predi, actuali], ["Predicted", "Actual"])
    # exit()
    indices = [4, 10, 33]
    images = [[top[i], side[i]] for i in indices]
    points = [[pred[i], actual[i]] for i in indices]
    labels = ["Predicted", "Actual"]
    titles = ["Top", "Side"]
    # make_error_figure(pred, actual)
    # make_2d_figure(images, points, labels, titles)
    # make_3d_heatmap_figure([predi, actuali], ["Predicted", "Actual"])


def main_2():
    from metrics import get_stats

    path = Path.cwd() / "data_3"

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

    top, side, actual, pred = load_data()

    # for i in range(len(pred)):
    #     reprojected_top_pred = point2pixel(pred[i], P_TOP)
    #     reprojected_top_actual = point2pixel(actual[i], P_TOP)
    #     plt.imshow(top[i])
    #     plt.scatter(
    #         reprojected_top_pred[:, 0],
    #         reprojected_top_pred[:, 1],
    #         label="Predicted"
    #     )
    #     plt.scatter(
    #         reprojected_top_actual[:, 0],
    #         reprojected_top_actual[:, 1],
    #         label="Actual"
    #     )
    #     plt.show()

    stats = get_stats(pred, actual)
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
