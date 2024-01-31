from argparse import ArgumentParser
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor

TOO_BIG = 12288


def set_vis_args(parser: ArgumentParser):
    parser.add_argument("-fmt", "--fig_format", type=str, default="png")
    parser.add_argument("-fs", "--fig_size", type=int, nargs=2, default=(9, 6))
    parser.add_argument("-mag", "--magnitude", type=str, default="m")
    parser.add_argument("-nb", "--num_bins", type=int, default=1024)
    parser.add_argument("-sort", "--sort_by", type=str, default="params")
    parser.add_argument("-xs", "--xscale", type=str, default="linear")
    parser.add_argument("-ys", "--yscale", type=str, default="linear")

    parser.add_argument("-o", "--output_dir", type=str, default="figures")


def plot_histogram(ax: Axes, hist: Tensor, edges: Tensor, label: str, color: Any):
    hist, edges = hist.cpu().numpy(), edges.cpu().numpy()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax.hist(
        edges[:-1],
        edges,
        weights=hist,
        edgecolor=color,
        label=label,
        linewidth=2,
        histtype="step",
        density=True,
    )
    plt.close()


def plot_graph(
    ax: Axes,
    points: Tensor,
    label: str,
    color: Any,
    intercept: Tuple[int, int] = None,
):
    points = points.cpu()
    ax.plot(points, label=label, color=color, markersize=0.5)
    if intercept:
        ax.axvline(intercept[0], color=color, linestyle="--", linewidth=0.2)
        ax.axhline(intercept[1], color=color, linestyle="--", linewidth=0.2)
    plt.close()


def plot_scatters(
    A: Tensor,
    B: Tensor,
    path: str,
    iden: str = None,
    label_A: str = "Classifiers",
    label_B: str = "Means",
    dims: Tuple[int, int] = (0, 1),
    intercept: Tuple[int, int] = None,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(B[0, :], B[1, :], alpha=0.25, c="r", label=label_B.lower())
    ax.scatter(A[0, :], A[1, :], alpha=0.25, c="b", label=label_A.lower())
    if intercept:
        ax.axvline(intercept[0], color="black", linestyle="--", linewidth=0.5)
        ax.axhline(intercept[1], color="black", linestyle="--", linewidth=0.5)

    ax.legend()
    ax.axis("equal")
    ax.set_xlabel(f"d={dims[0]}")
    ax.set_ylabel(f"d={dims[1]}")

    title = f"{label_A} vs. {label_B}"
    if iden:
        title = f"{title} ({iden})"
    ax.set_title(title)
    fig.savefig(path)

    plt.close()


def visualize_matrix(data: Tensor, mat_path: str, figsize=(6, 6)):
    plt.matshow(data.cpu())
    plt.colorbar()
    plt.savefig(mat_path)
    plt.close()


def get_shade(color: str, index: int, total: int, floor: float = 0.65) -> str:
    r, g, b = [int(color[i : i + 2], 16) for i in (1, 3, 5)]

    brightness = floor + (total - index) * (1 - floor) / total
    r, g, b = [int(c * brightness) for c in (r, g, b)]

    shade = f"#{r:02x}{g:02x}{b:02x}"
    return shade
