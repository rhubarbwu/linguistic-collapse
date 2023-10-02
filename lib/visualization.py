from argparse import ArgumentParser, Namespace
from typing import Any, Tuple

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
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
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
        ax.axvline(intercept[0], color=color, linestyle="--", linewidth=0.5)
        ax.axhline(intercept[1], color=color, linestyle="--", linewidth=0.5)
