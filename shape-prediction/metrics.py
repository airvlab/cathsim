import numpy as np
from collections import OrderedDict

from scipy.spatial.distance import (
    cdist,
    directed_hausdorff,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


class Metric:
    def __init__(
        self,
        name: str,
        func: callable,
        unit: str = None,
        agg_func: callable = None,
    ):
        self.name = name
        self.func = func
        self.value = None
        self.unit = unit
        self.agg_func = agg_func

    def __repr__(self):
        repr = f"{self.name}: {self.value:.3f}"
        if self.std_dev is not None:
            repr += f" ± {self.std_dev:.3f}"
        if self.unit is not None:
            repr += f" {self.unit}"

    def compute(self, pred, actual):
        self.value = self.func(pred, actual)
        if self.agg_func is not None:
            self.value, self.std_dev = self.agg_func(self.value)


def get_stats(pred: list, actual: list, metrics: dict = None, round: int = 3) -> dict:
    """Get statistics for the predicted points.

    The function computes a set of metrics given to sets of points.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.
        metrics (dict): Dictionary of metrics to compute.
    """
    if metrics is None:
        metrics = METRICS

    stats = {}
    for name, metric in metrics.items():
        result = []
        for i in range(len(pred)):
            result.append(metric(pred[i], actual[i]))

        stats[name] = np.round(np.mean(result) * 1000, round)
    return stats


def mse(pred, actual):
    """
    Mean Squared Error (MSE)

    Description:
        Measures the average of the squares of the differences between predicted and actual points.

    Interpretation:
        Lower values are better. A value of 0 indicates a perfect match. Gives more weight to larger errors.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The mean squared error value.
    """
    return mean_squared_error(pred, actual)


def mae(pred, actual):
    """
    Mean Absolute Error (MAE)

    Description:
        Measures the average of the absolute differences between predicted and actual points.

    Interpretation:
        Lower values are better. A value of 0 indicates a perfect match. Less sensitive to extreme values compared to MSE.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The mean absolute error value.
    """
    return np.mean(np.abs(pred - actual))


def hausdorff_distance(pred, actual):
    """
    Hausdorff Distance

    Description:
        Gives the maximum distance of a point in a predicted set to the nearest point in the actual set.

    Interpretation:
        Lower values are better. A value of 0 indicates a perfect match. Useful for understanding the worst-case deviation.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The Hausdorff distance value.
    """
    return max(directed_hausdorff(pred, actual)[0], directed_hausdorff(actual, pred)[0])


def med(pred, actual):
    """
    Mean Euclidean Distance (MED)

    Description:
        Gives an average distance between corresponding points in two sets.

    Interpretation:
        Lower values are better. A value of 0 indicates the two sets perfectly overlap.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The mean Euclidean distance.
    """
    return np.mean(np.linalg.norm(pred - actual, axis=1))


def maxed(pred, actual):
    """
    Maximum Euclidean Distance (MaxED)

    Description:
        Indicates the largest deviation between two corresponding points.

    Interpretation:
        Lower values are better. A value of 0 indicates the two sets perfectly overlap at all points.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The maximum Euclidean distance.
    """
    return np.max(np.linalg.norm(pred - actual, axis=1))


def sdd(pred, actual):
    """
    Standard Deviation of Distances (SDD)

    Description:
        Measures the spread of distances between the points of the two sets.

    Interpretation:
        Lower values suggest a more consistent error pattern, whereas higher values suggest more variability.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The standard deviation of distances.
    """
    distances = np.linalg.norm(pred - actual, axis=1)
    return np.std(distances)


def segment_cosine_similarity(pred, actual):
    """
    Segment Cosine Similarity

    Description:
        Measures the cosine similarity between corresponding segments in the predicted and actual sets.

    Interpretation:
        Values closer to 1 indicate a high similarity in orientation. A value of -1 indicates completely opposite directions.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The average cosine similarity of segments.
    """
    pred_directions = np.diff(pred, axis=0)
    actual_directions = np.diff(actual, axis=0)
    similarities = [
        cosine_similarity(p_dir.reshape(1, -1), a_dir.reshape(1, -1)).item()
        for p_dir, a_dir in zip(pred_directions, actual_directions)
    ]
    return np.mean(similarities)


def mean_angle_difference(pred, actual):
    """
    Mean Angle Difference

    Description:
        Measures the average difference in angles between corresponding segments of the predicted and actual sets.

    Interpretation:
        Lower values indicate a closer match in orientation. A value of 0 indicates the segments have the same direction.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The average difference in angles.
    """
    pred_directions = np.diff(pred, axis=0)
    actual_directions = np.diff(actual, axis=0)

    pred_angles = np.arctan2(pred_directions[:, 1], pred_directions[:, 0])
    actual_angles = np.arctan2(actual_directions[:, 1], actual_directions[:, 0])

    angle_diffs = np.abs(pred_angles - actual_angles)
    return np.mean(angle_diffs)


def cumulative_distance(pred, actual):
    """
    Cumulative Distance

    Description:
        Sum of the Euclidean distances between corresponding points of two curves.

    Interpretation:
        Lower values are better. A value of 0 indicates the two curves perfectly overlap.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The cumulative distance.
    """
    return np.sum(np.linalg.norm(pred - actual, axis=1))


def curve_length_difference(pred, actual):
    """
    Curve Length Difference

    Description:
        Difference between the lengths of the two curves.

    Interpretation:
        A value of 0 indicates both curves are of the same length. Positive values indicate the predicted curve is longer.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The difference in lengths of two curves.
    """
    pred_length = np.sum(np.linalg.norm(np.diff(pred, axis=0), axis=1))
    actual_length = np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=1))

    return abs(pred_length - actual_length)


def dice_coefficient(pred, actual, threshold=0.5):
    """
    Dice Similarity Coefficient (Volume Overlap)

    Description:
        A statistical measure for the overlap between two binary volumes.

    Interpretation:
        Values range between 0 (no overlap) and 1 (perfect overlap). Values closer to 1 indicate a better reconstruction.

    Args:
        pred (np.ndarray): Predicted points, thresholded into binary values.
        actual (np.ndarray): Actual points, thresholded into binary values.

    Returns:
        float: The Dice coefficient indicating the similarity between two volumes.
    """
    pred_binary = (pred > threshold).astype(np.int32)
    actual_binary = (actual > threshold).astype(np.int32)

    intersection = np.sum(pred_binary * actual_binary)
    union = np.sum(pred_binary) + np.sum(actual_binary)

    return (2.0 * intersection) / (union + 1e-10)


def euclidean_distance(pt1, pt2):
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def calculate_frechet_distance(P, Q, i, j, memo):
    """
    Recursive function to calculate discrete Fréchet distance between two curves P and Q.
    """
    if (i, j) in memo:
        return memo[(i, j)]
    elif i == len(P) - 1 and j == len(Q) - 1:
        memo[(i, j)] = euclidean_distance(P[i], Q[j])
    elif i == len(P) - 1:
        memo[(i, j)] = max(
            euclidean_distance(P[i], Q[j]),
            calculate_frechet_distance(P, Q, i, j + 1, memo),
        )
    elif j == len(Q) - 1:
        memo[(i, j)] = max(
            euclidean_distance(P[i], Q[j]),
            calculate_frechet_distance(P, Q, i + 1, j, memo),
        )
    else:
        dist_current = euclidean_distance(P[i], Q[j])
        dist_next = min(
            calculate_frechet_distance(P, Q, i + 1, j, memo),
            calculate_frechet_distance(P, Q, i, j + 1, memo),
            calculate_frechet_distance(P, Q, i + 1, j + 1, memo),
        )
        memo[(i, j)] = max(dist_current, dist_next)

    return memo[(i, j)]


def frechet_distance(pred, actual):
    """
    Discrete Fréchet Distance

    Description:
        It quantifies the similarity between two curves in a metric space.

    Interpretation:
        Lower values indicate that the two curves are closer to each other.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The Fréchet distance between two curves.
    """
    memo = {}
    return calculate_frechet_distance(pred, actual, 0, 0, memo)


def mean_tip_displacement(pred, actual):
    """
    Mean Tip Displacement

    Description:
        Measures the average distance between the tips of the predicted and actual sets.

    Interpretation:
        Lower values are better. A value of 0 indicates a perfect match.

    Args:
        pred (np.ndarray): Predicted points.
        actual (np.ndarray): Actual points.

    Returns:
        float: The mean tip displacement.
    """
    return np.mean(np.linalg.norm(pred[0] - actual[0]))


METRICS = OrderedDict(
    mse=mse,
    # hausdorff_distance=hausdorff_distance,
    # mae=mae,
    # med=med,
    maxed=maxed,
    # sdd=sdd,
    # segment_cosine_similarity=segment_cosine_similarity,
    # mean_angle_difference=mean_angle_difference,
    # cumulative_distance=cumulative_distance,
    # curve_length_difference=curve_length_difference,
    # dice_coefficient=dice_coefficient,
    # frechet_distance=frechet_distance,
    mean_tip_displacement=mean_tip_displacement,
)
