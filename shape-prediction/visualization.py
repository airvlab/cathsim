import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import cv2
import trimesh
from cathsim.dm.visualization import point2pixel
from scipy.spatial import cKDTree


P_TOP = np.array(
    [
        [-5.79411255e02, 0.00000000e00, 2.39500000e02, -7.72573376e01],
        [0.00000000e00, 5.79411255e02, 2.39500000e02, -1.32301407e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00, -2.50000000e-01],
    ]
)

P_SIDE = np.array(
    [
        [-2.39500000e02, 5.79411255e02, 0.00000000e00, -1.13528182e02],
        [-2.39500000e02, 0.00000000e00, 5.79411255e02, -7.00723376e01],
        [-1.00000000e00, 0.00000000e00, 0.00000000e00, -2.20000000e-01],
    ]
)


def visualize_images(top_images, side_images, n=2):
    fig, ax = plt.subplots(n, 2)
    for i in range(n):
        ax[i][0].imshow(top_images[i], cmap="gray")
        ax[i][0].axis("off")
        ax[i][1].imshow(side_images[i], cmap="gray")
        ax[i][1].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_projections(
    top_images: list[np.ndarray],
    side_images: list[np.ndarray],
    points: list[list[np.ndarray]],
    n: int = 2,
):
    fig, ax = plt.subplots(n, 2)
    for i in range(n):
        reprojection_top = point2pixel(points[i], P_TOP)
        reprojection_side = point2pixel(points[i], P_SIDE)

        if n == 1:
            ax[0].imshow(top_images[i], cmap="gray")
            ax[0].scatter(reprojection_top[:, 0], reprojection_top[:, 1], s=0.3)
            ax[0].axis("off")
            ax[1].imshow(side_images[i], cmap="gray")
            ax[1].scatter(reprojection_side[:, 0], reprojection_side[:, 1], s=0.3)
            ax[1].axis("off")
            continue

        ax[i][0].imshow(top_images[i], cmap="gray")
        ax[i][0].scatter(reprojection_top[:, 0], reprojection_top[:, 1], s=0.3)
        ax[i][0].axis("off")
        ax[i][1].imshow(side_images[i], cmap="gray")
        ax[i][1].scatter(reprojection_side[:, 0], reprojection_side[:, 1], s=0.3)
        ax[i][1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_curve_on_image(ax, curve):
    cs_x, cs_y, t = curve
    t_new = np.linspace(0, t[-1], 1000)
    curve_points_x = cs_x(t_new)
    curve_points_y = cs_y(t_new)

    ax.plot(curve_points_x, curve_points_y, "r-", linewidth=1)


def set_limits(ax, plot_mesh=True):
    mesh_path = Path.cwd() / "assets/visual.stl"
    if not mesh_path.exists():
        raise FileNotFoundError("Visual mesh not found")
    mesh = trimesh.load_mesh(mesh_path)
    if plot_mesh:
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            alpha=0.1,
            color="gray",
        )
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])


def visualize_points_2d(images: list, point_sets: list):
    images = [image.copy() for image in images]
    for i in range(len(images)):
        for point in point_sets[i]:
            cv2.circle(images[i], tuple(point.astype(int)), 1, (0, 255, 0), -1)
    combined = np.concatenate(images, axis=1)
    cv2.imshow("combined", combined)
    cv2.waitKey(1)


def visualize_2d(
    images: tuple[np.ndarray], point_sets: tuple[list[np.ndarray]], labels: list
):
    fig, ax = plt.subplots(1, len(images))
    for i, (image, points, labels) in enumerate(zip(images, point_sets, labels)):
        for p in points:
            ax[i].plot(p[:, 0], p[:, 1], ".", label=labels)
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.1, 1.1))
    plt.show()


def visualize_3d(point_sets: list, labels: list, colors: list = ["r", "g"]):
    fig = plt.figure(figsize=(4, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    for i, (points, label) in enumerate(zip(point_sets, labels)):
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            ".",
            label=label,
            color=colors[i % len(colors)]
        )

    set_limits(ax, plot_mesh=False)

    ax.view_init(elev=0, azim=180)
    # make tick labels invisible
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    fig.legend(
        ncol=len(labels),
        loc="outside upper center",
        markerscale=5,
        # text size
        prop={"size": 8},
    )
    # fig.savefig(
    #     "./data/figures/3d.jpg",
    #     dpi=300 * 4,
    #     bbox_inches="tight",
    #     pad_inches=0,
    # )
    plt.show()


def visualize_curves_2d(images: list, curves: list):
    fig, ax = plt.subplots(1, len(images))
    for i, (image, curve) in enumerate(zip(images, curves)):
        plot_curve_on_image(ax[i], curve)
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.1, 1.1))
    plt.show()


def visualize_points_2d_matplotlib(
    images: list, point_sets: list, labels: list, titles: list
):
    fig, ax = plt.subplots(1, len(images))
    images = [image.copy() for image in images]
    for i in range(len(images)):
        for j, point in enumerate(point_sets[i]):
            ax[i].plot(point[:, 0], point[:, 1], ".", label=labels[j])
        ax[i].imshow(images[i])
        ax[i].axis("off")
        ax[i].set_title(titles[i])
    handles, labels = ax[0].get_legend_handles_labels()
    # reduce the space between legend and the figure
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        # loc="upper center",
        markerscale=5,
        # frameon=True,
        # bbox_to_anchor=(0.26, 1, 1, 0),
        loc="outside upper center",
        # mode="expand",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    # fig.savefig("./data/figures/reprojection.jpg", dpi=300)
    plt.show()


def filter_data_within_limits(points, xlim, ylim, zlim):
    mask = (
        (points[:, 0] > xlim[0])
        & (points[:, 0] < xlim[1])
        & (points[:, 1] > ylim[0])
        & (points[:, 1] < ylim[1])
        & (points[:, 2] > zlim[0])
        & (points[:, 2] < zlim[2])
    )
    return points[mask]


def zoom_and_crop(ax, x, y, z, zoom_factor):
    x_range = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2.0
    y_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2.0
    z_range = (ax.get_zlim()[1] - ax.get_zlim()[0]) / 2.0

    xlim = [x - x_range * zoom_factor, x + x_range * zoom_factor]
    ylim = [y - y_range * zoom_factor, y + y_range * zoom_factor]
    zlim = [z - z_range * zoom_factor, z + z_range * zoom_factor]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    return xlim, ylim, zlim


def make_2d_figure(
    images: list, point_sets: list, labels: list, titles: list, alpha=0.7
):
    def save_fig(fig):
        fig.savefig(
            "./assets/figures/actual_vs_pred_2d.png",
            dpi=300 * 4,
            bbox_inches="tight",
            pad_inches=0,
        )

    cols = len(images)
    rows = len(images[0])

    fig, ax = plt.subplots(
        rows,
        cols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(5.5, 4),
    )
    fig.subplots_adjust(hspace=0.02, wspace=0.02)

    for j in range(cols):
        for i in range(rows):
            image = images[j][i]
            pred_points = point_sets[j][0]
            actual_points = point_sets[j][1]

            if i == 0:
                P = P_TOP
            else:
                P = P_SIDE

            reprojection_pred = point2pixel(pred_points, P)
            reprojection_actual = point2pixel(actual_points, P)

            ax[i][j].imshow(image, cmap="gray")

            ax[i][j].plot(
                reprojection_pred[:, 0],
                reprojection_pred[:, 1],
                label=labels[0],
                lw=1,
                alpha=alpha,
            )

            ax[i][j].plot(
                reprojection_actual[:, 0],
                reprojection_actual[:, 1],
                label=labels[1],
                lw=1,
                alpha=alpha,
            )

            ax[i][j].axis("off")
            ax[i][j].set_aspect("equal")
            if j == 0:
                if i == 0:
                    ax[i][j].annotate(
                        "Top View",
                        xy=(-0.2, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="center",
                        rotation="vertical",
                        fontsize=10,
                    )
                else:
                    ax[i][j].annotate(
                        "Side View",
                        xy=(-0.2, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="center",
                        rotation="vertical",
                        fontsize=10,
                    )

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="outside upper center",
        markerscale=20,
    )
    save_fig(fig)
    plt.show()


# Ensure P_TOP and P_SIDE are defined elsewhere in your code or pass them as parameters.


def make_3d_figure(point_sets: list, labels: list):
    def save_fig(fig):
        fig.savefig(
            "./assets/figures/actual_vs_pred_3d.png",
            dpi=300 * 2,
            bbox_inches="tight",
            pad_inches=0,
        )

    views = [
        # (0, 180),
        (0, 0),
        # (90, 0),
    ]

    fig = plt.figure(figsize=(4, 3.0))
    for i, view in enumerate(views):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        for points, label in zip(point_sets, labels):
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                # ".",
                label=label,
                linewidth=1,
                # markersize=2,
            )
            set_limits(ax, plot_mesh=True)
            # ax.view_init(elev=view[0], azim=view[1])

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            # zoom_and_crop(ax, 0.8)

    fig.legend(
        ncol=2,
        loc="outside upper center",
        markerscale=5,
        # text size
        prop={"size": 8},
        # bbox_to_anchor=(
        #     -0.0,
        #     0.5,
        # ),
    )
    # save_fig(fig)
    # plt.show()
    #
    #


def make_3d_heatmap_figure(point_sets: list, labels: list):
    def save_fig(fig):
        fig.savefig(
            "./assets/figures/actual_vs_pred_3d_heatmap.png",
            dpi=300 * 4,
            bbox_inches="tight",
            pad_inches=0,
        )

    def make_colorbar(fig, scatter_plot, cbar_min_dist, cbar_max_dist):
        cbar_position = [0.87, 0.15, 0.02, 0.6]
        cbar_ax = fig.add_axes(cbar_position)
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Error Distance")

        min_tick = norm(cbar_min_dist)
        max_tick = norm(cbar_max_dist)

        cbar.set_ticks([min_tick, max_tick])
        cbar.set_ticklabels(
            [f"{cbar_min_dist*1000:.2f}mm", f"{cbar_max_dist*1000:.2f}mm"]
        )

    points_reconstructed = point_sets[0]
    points_actual = point_sets[1]
    print(points_reconstructed.shape, points_actual.shape)

    kdtree = cKDTree(points_actual)
    distances, indices = kdtree.query(points_reconstructed)
    norm = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    colors = plt.cm.viridis(norm(distances))

    views = [
        (0, 180),
        (25, -25),
    ]
    cbar_max_height = 0

    fig = plt.figure(figsize=(5.5, 3.0))

    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1.3, 0.5], wspace=0)

    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[i], projection="3d")

        scatter_plot = ax.scatter(
            points_reconstructed[:, 0],
            points_reconstructed[:, 1],
            points_reconstructed[:, 2],
            c=colors,
            label=labels[0],
            s=1,
        )

        ax.plot(
            points_actual[:, 0],
            points_actual[:, 1],
            points_actual[:, 2],
            label=labels[1],
            linewidth=1,
            markersize=2,
        )

        set_limits(ax, plot_mesh=False)

        ax.view_init(elev=view[0], azim=view[1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        cbar_max_height = max(cbar_max_height, np.max(distances))

        if i == 0:
            zoom_and_crop(ax, 0.0, 0.08, 0.01, 0.5)
        else:
            zoom_and_crop(ax, 0.035, 0.03, 0.04, 0.3)

    # Define the colorbar axis based on the third column of the GridSpec
    # cbar_ax = plt.subplot(gs[2])
    make_colorbar(fig, scatter_plot, np.min(distances), np.max(distances))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        markerscale=5,
        prop={"size": 8},
    )
    save_fig(fig)
    plt.show()


def compute_pointwise_errors(pred_list, actual_list):
    """Compute errors for each point across all samples."""
    num_points = min(
        len(pred_list[0]), len(actual_list[0])
    )  # assuming each sample in a list has the same number of points
    pointwise_errors = [[] for _ in range(num_points)]

    for pred, actual in zip(pred_list, actual_list):
        for i in range(num_points):
            error = np.linalg.norm(pred[i] - actual[i])
            pointwise_errors[i].append(error)
    return pointwise_errors


def make_error_figure(pred_list, actual_list):
    def save_fig(fig):
        fig.savefig(
            "./assets/figures/error_plot.png",
            dpi=300 * 4,
            bbox_inches="tight",
            pad_inches=0,
        )

    for i, (pred, actual) in enumerate(zip(pred_list, actual_list)):
        assert len(pred) == len(actual)
    pointwise_errors = compute_pointwise_errors(pred_list, actual_list)

    mean_errors = [np.mean(errors) for errors in pointwise_errors]
    std_errors = [np.std(errors) for errors in pointwise_errors]

    fig, ax = plt.subplots(figsize=(4.8, 2.0))

    x = np.arange(len(mean_errors))  # point locations
    ax.plot(
        x,
        mean_errors,
        "-",
        color="black",
    )
    ax.errorbar(
        x,
        mean_errors,
        yerr=std_errors,
        fmt="o",
        # color="blue",
        ecolor="black",
        markersize=3,
        capsize=3,
    )

    ax.set_xlabel("Point")
    ax.set_yticklabels([f"{tick*1000:.2f}" for tick in ax.get_yticks()])
    ax.set_ylabel("Error (mm)")
    ax.set_title("Error Along the Guidewire")
    ax.set_xticks([i for i in x if i % 10 == 0])
    ax.set_xticklabels([i for i in x if i % 10 == 0])
    ax.yaxis.grid(True)

    plt.tight_layout()
    save_fig(fig)
    plt.show()


def make_simple_3d_figure(points):
    def save_fig(fig):
        fig.savefig(
            "./assets/figures/simple_3d.png",
            dpi=300 * 2,
            bbox_inches="tight",
            pad_inches=0,
        )

    fig = plt.figure(figsize=(4, 3.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        # ".",
        color="r",
        linewidth=2,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        ".",
        color="r",
        s=20,
    )

    set_limits(ax, plot_mesh=True)
    # set the view
    # ax.view_init(elev=45, azim=45)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    save_fig(fig)
    plt.show()
