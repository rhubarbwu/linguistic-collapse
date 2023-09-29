from typing import Any, Tuple

from matplotlib.axes import Axes
from torch import Tensor

TOO_BIG = 12288


def plot_histogram(ax: Axes, hist: Tensor, edges: Tensor, label: str, color: Any):
    hist, edges = hist.cpu(), edges.cpu()
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
