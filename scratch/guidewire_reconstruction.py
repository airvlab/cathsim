import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from time import sleep
import numpy.linalg as la


P_top = np.array(
    [[5.79411255e02, 0.00000000e00, 2.40000000e02, -4.26176624e01],
     [0.00000000e00, 5.79411255e02, 2.40000000e02, -1.32426407e02],
     [0.00000000e00, 0.00000000e00, 1.00000000e00, -2.50000000e-01]]
)

P_side = np.array(
    [[3.54787069e-14, -2.40000000e02, 5.79411255e02, 1.20270476e02],
     [-5.79411255e02, -2.40000000e02, 5.01744685e-14, -6.80381818e01],
     [0.00000000e00, -1.00000000e00, 6.12323400e-17, -3.00000000e-02]]
)


def load_data(path: Path):
    return zip(sorted(list(path.glob('*_top.npy')), key=lambda x: x.name), sorted(list(path.glob('*_side.npy')), key=lambda x: x.name), sorted(list(path.glob('*_geom_pos.npy')), key=lambda x: x.name))


def load_image(path: Path):
    return np.load(path)[:, :, 0]


def load_data_from_paths(data: Tuple[Path, Path, Path]):
    return load_image(data[0]), load_image(data[1]), np.load(data[2])


def get_segmented_centroids(image, segment_height_pixel):
    height, width = image.shape
    centroids = []

    for i in range(0, height, max(1, segment_height_pixel)):
        segment = image[i: i + segment_height_pixel, :]
        contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cY += i
                centroids.append([cX, cY])

    return np.array(centroids)


def real_to_pixel_length_from_matrix(P, distance_to_object_m, actual_length_m):
    return int((P[0, 0] / distance_to_object_m) * actual_length_m)


def get_camera_pos_from_matrix(P):
    return -la.inv(P[:, :3]) @ P[:, 3]


def d2tip(P, head_pos):
    return la.norm(get_camera_pos_from_matrix(P) - head_pos)


def triangulate_points(P1, P2, points1, points2):
    num_points = points1.shape[0]
    points1_homogeneous = np.hstack([points1, np.ones((num_points, 1))])
    points2_homogeneous = np.hstack([points2, np.ones((num_points, 1))])

    points_3d = cv2.triangulatePoints(P1, P2, points1_homogeneous.T, points2_homogeneous.T)
    points_3d /= points_3d[3]
    return points_3d[:3].T


if __name__ == "__main__":
    path = Path.cwd() / 'data' / 'guidewire-reconstruction'
    data = load_data(path)

    all_3d_points = []

    for d in data:
        top, side, shape = load_data_from_paths(d)
        if (top == 0).all() or (side == 0).all():
            continue
        head_pos = shape[-1]
        distance_to_object_m_top = d2tip(P_top, head_pos)
        distance_to_object_m_side = d2tip(P_side, head_pos)

        print(f"Distance to object in meters (top): {distance_to_object_m_top}")
        print(f"Distance to object in meters (side): {distance_to_object_m_side}")

        segment_height_m = 0.002
        segment_height_pixel_top = real_to_pixel_length_from_matrix(P_top, distance_to_object_m_top, segment_height_m)
        segment_height_pixel_side = real_to_pixel_length_from_matrix(P_side, distance_to_object_m_side, segment_height_m)

        print(f"Segment height in pixels (top): {segment_height_pixel_top}")
        print(f"Segment height in pixels (side): {segment_height_pixel_side}")

        centroids_top = get_segmented_centroids(top, segment_height_pixel_top)
        centroids_side = get_segmented_centroids(side, segment_height_pixel_side)

        print(f"Number of centroids (top): {centroids_top.shape[0]}")
        print(f"Number of centroids (side): {centroids_side.shape[0]}")

        if centroids_top.shape[0] == centroids_side.shape[0]:
            points_3d = triangulate_points(P_top, P_side, centroids_top, centroids_side)
            all_3d_points.append(points_3d)

    exit()

    for d in data:
        top, side, shape = load_data_from_paths(d)
        cv2.imshow('top', top)
        cv2.imshow('side', side)
        cv2.waitKey(1)
        sleep(1)
