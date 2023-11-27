from argparse import ArgumentParser
from os import makedirs
from os.path import exists, isfile
from random import sample
from typing import Tuple

import matplotlib.pyplot as plt
import torch as pt

from lib.collapse import Statistics
from lib.model import (COLOUR_BASES, DEPTHS, get_classifier_weights,
                       get_model_colour, split_parts)
from lib.statistics import collect_hist, meanify_diag
from lib.utils import MAG_STRS, MAGS, identify, inner_product
from lib.visualization import (TOO_BIG, plot_graph, plot_histogram,
                               plot_scatters, set_vis_args, visualize_matrix)

pt.set_grad_enabled(False)

LINE_SEP = "-" * 79


parser = ArgumentParser()

parser.add_argument("-dev", "--device", type=str, default="cpu")
parser.add_argument(
    "-total",
    "--totals",
    type=int,
    nargs=3,
    default=(469514249, 229367, 29233),
)

parser.add_argument("-i", "--input_files", type=str, nargs="+", default=[])
parser.add_argument("-mc", "--model_cache", type=str, default=".")

parser.add_argument("-prog", "--progress", action="store_true")
parser.add_argument("-eig", "--eigenvalues", action="store_true")
parser.add_argument("-nor", "--norms", action="store_true")
parser.add_argument("-coh", "--coherence", action="store_true")
parser.add_argument("-dual", "--duality", action="store_true")
parser.add_argument("-snr", "--inv_snr", action="store_true")
parser.add_argument("-all", "--analysis", action="store_true")
parser.add_argument("-each", "--each_model", action="store_true")

parser.add_argument("-mpc", "--min_per_class", type=int, default=1)
parser.add_argument("-Mpc", "--max_per_class", type=int, default=None)
parser.add_argument("-ps", "--patch_size", type=int, default=1024)
parser.add_argument("-d", "--dims", type=int, nargs=2, default=None)
parser.add_argument("-split", "--split_char", type=str, default="x")

set_vis_args(parser)
args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"W: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"

SORT_WIDTH = "dim" in args.sort_by or "wid" in args.sort_by

if args.analysis:
    args.eigenvalues = args.norms = args.coherence = args.inv_snr = args.duality = True
args.analysis |= args.eigenvalues | args.norms | args.coherence
args.analysis |= args.duality | args.inv_snr


PATHS = {}
for file in args.input_files:
    if not exists(file) or not isfile(file):
        continue
    iden = identify(file)
    if iden not in PATHS:
        PATHS[iden] = (None, None)

    means_path = file if "means" in file else PATHS[iden][0]
    vars_path = file if "vars" in file else PATHS[iden][1]
    PATHS[iden] = (means_path, vars_path)


if args.progress:
    COL_WIDTH = 6
    PROGRESS = {}
if args.analysis:
    STATISTICS = {}

INCOMPLETE = []
for iden, (means_path, vars_path) in PATHS.items():
    collected = Statistics(
        device=args.device,
        load_means=means_path,
        load_vars=vars_path,
        verbose=False,
    )
    Ns = (collected.N1, collected.N1_seqs, collected.N2, collected.N2_seqs)
    if Ns[0] != args.totals[0] or Ns[1] != args.totals[1]:
        INCOMPLETE.append(iden)

    if args.analysis:
        STATISTICS[iden] = collected

    if args.progress:
        PROGRESS[iden] = (*Ns, collected.counts_in_range(args.min_per_class).shape[0])
        COL_WIDTH = max(COL_WIDTH, max(len(str(n)) for n in PROGRESS[iden]))

    del collected


IDENTIFIERS = sorted(PATHS.keys(), key=lambda x: split_parts(x)[1 if SORT_WIDTH else 0])
LONGEST_IDEN = max(5, max([len(iden) for iden in IDENTIFIERS]))

if args.progress:
    print(LINE_SEP)
    head = [p.rjust(COL_WIDTH) for p in ["means", "(seqs)", "vars", "(seqs)", "unique"]]
    print("model".ljust(LONGEST_IDEN + 1), *head)
    for iden in IDENTIFIERS:
        Ns = PROGRESS[iden]
        row = [str(n).rjust(COL_WIDTH) for n in Ns]
        print(iden.ljust(LONGEST_IDEN + 1), *row)
    row = [str(n).rjust(COL_WIDTH) for n in args.totals[:2] + args.totals]
    print("total".ljust(LONGEST_IDEN + 1), *row)


print(LINE_SEP)
if not args.analysis:
    exit()


for iden in INCOMPLETE:
    del STATISTICS[iden]
    IDENTIFIERS.remove(iden)


ANALYTICS = {}
BY_DEPTH, BY_WIDTH = {depth: [] for depth in DEPTHS}, {}
for iden in IDENTIFIERS:
    if iden not in STATISTICS:
        continue

    for prop_str, group in zip(split_parts(iden)[:2], [BY_DEPTH, BY_WIDTH]):
        prop_val = int(prop_str)
        if prop_val not in group:
            group[prop_val] = []
        group[prop_val].append(iden)

    collected: Statistics = STATISTICS[iden]
    indices = collected.counts_in_range(args.min_per_class, args.max_per_class)
    counts = collected.counts[indices]
    means, mean_G = collected.compute_means(indices)
    if args.dims is None:
        args.dims = sample(list(range(len(indices))), 2)

    if iden not in ANALYTICS:
        ANALYTICS[iden] = {}

    if args.eigenvalues:
        outer = inner_product(means.T, args.patch_size, "eigs outer")
        if outer.shape[0] > TOO_BIG:
            print(f"W: {outer.shape[0]} > {TOO_BIG}, eigenvalues disabled")
        else:
            eig_vals = pt.linalg.eigvalsh(outer)
            ANALYTICS[iden]["eig_vals"] = eig_vals.real

    if args.norms:
        norms = means.norm(dim=-1) ** 2
        norms_hist = collect_hist(norms.unsqueeze(0), args.num_bins, desc="norms hist")
        ANALYTICS[iden]["norms_cv"] = norms.std() / norms.mean()
        ANALYTICS[iden]["norms_hist"] = norms_hist
        ANALYTICS[iden]["freqs_norms"] = pt.stack((counts, norms))
        del norms

    if args.coherence:
        inner = collected.coherence(indices, args.patch_size)
        hist, edges = collect_hist(inner, args.num_bins, True, desc="coh hist")
        ANALYTICS[iden]["coh_hist"] = (hist, edges)

        inner = meanify_diag(inner)
        ANALYTICS[iden]["coh_mean"] = inner.mean().cpu()
        ANALYTICS[iden]["coh_std"] = inner.std().cpu()
        ANALYTICS[iden]["coh_cv"] = inner.std().cpu() / inner.mean().cpu()
        del inner, hist, edges

    if args.duality:
        args.model_name = f"TS{iden}"
        W = get_classifier_weights(args)
        if W is not None:
            dual_diff = collected.diff_duality(W, indices)
            ANALYTICS[iden]["dual_diff"] = dual_diff.mean().cpu()

            dual_dot, proj_m, proj_c = collected.dot_duality(W, indices, args.dims)
            ANALYTICS[iden]["dual_dot_projs"] = proj_m, proj_c
            ANALYTICS[iden]["dual_dot"] = dual_dot.mean().cpu()

            dual_dot_hist = collect_hist(dual_dot, args.num_bins, desc="dual hist")
            ANALYTICS[iden]["dual_hist"] = dual_dot_hist
            del W

    if args.inv_snr:
        CDNVs = collected.compute_vars(indices)
        if CDNVs is not None and collected.N2 == args.totals[0]:
            meanify_diag(CDNVs)
            CDNVs = pt.log(CDNVs)
            # CDNVs = CDNVs[CDNVs > 1e-4].unsqueeze(0)

            ANALYTICS[iden]["inv_snr"] = CDNVs.std().cpu() / CDNVs.mean().cpu()
            ANALYTICS[iden]["inv_snr_hist"] = collect_hist(CDNVs, args.num_bins, True)
            if args.each_model:
                matpath = f"{args.output_dir}/cdnv-{iden}.{args.fig_format}"
                visualize_matrix(CDNVs, matpath)
            del CDNVs

    del collected


## VISUALIZATION ##


makedirs(args.output_dir, exist_ok=True)


range_str = ""
if args.min_per_class:
    if args.max_per_class:
        range_str = f" ({args.min_per_class} \u2264 counts < {args.max_per_class})"
    else:
        range_str = f" (counts \u2265 {args.min_per_class})"


def plot_statistic(measure: str, key: str):
    fig, ax = plt.subplots(figsize=args.fig_size)

    for depth, values in sorted(BY_DEPTH.items()):
        nums, vals = [], []
        for iden in values:
            if iden in ANALYTICS and key in ANALYTICS[iden]:
                nums.append(split_parts(iden)[1 if SORT_WIDTH else 0])
                vals.append(ANALYTICS[iden][key])

        ax.scatter(
            *(nums, vals),
            c=COLOUR_BASES[DEPTHS.index(depth)],
            marker="o",
            label=f"n_layers={depth}",
        )

        for x, y, label in zip(nums, vals, values):
            ax.annotate(label, (x, y))

    ax.legend()
    ax.set_ylabel(measure.lower())
    if SORT_WIDTH:
        ax.set_title(f"{measure} vs. Model Size{range_str}")
        ax.set_xlabel(f"Hidden Dimension")
    else:
        ax.set_title(f"{measure} vs. Model Dimension{range_str}")
        ax.set_xlabel(f"Parameter Count ({MAG_STRS[MAGS.index(args.magnitude)]}s)")

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/{key}.{args.fig_format}")
    plt.close()


def plot_arrays(
    title: str,
    measure: str,
    xlabel: str = "[xlabel]",
    ylabel: str = "[ylabel]",
    xscale: str = None,
    yscale: str = None,
    selected: str = None,
):
    fig, ax = plt.subplots(figsize=args.fig_size)
    for iden in [selected] if selected else IDENTIFIERS:
        array = ANALYTICS[iden][measure]
        if array is None:
            continue

        label, color = iden, get_model_colour(
            *split_parts(iden)[:2],
            list(BY_DEPTH.keys()),
            list(BY_WIDTH.keys()),
            SORT_WIDTH,
        )
        if "hist" in measure:
            plot_histogram(ax, *array, label, color)
            continue

        array = array.cpu()
        if len(array.shape) == 1:
            intercept = (len(array) - 1, array[-1])
            # intercept = None
            plot_graph(ax, array, label, color, intercept)
        else:
            ax.scatter(*array, 5, color, alpha=0.15, label=label)

    if "hist" in measure:
        xlabel, ylabel = f"Value ({args.num_bins} bins)", "Frequency"
    elif "eig" in measure:
        xlabel, ylabel = "Dimension", "Eigenvalue"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} {'('+selected+') ' if selected else ''}{range_str}")
    ax.set_xscale(xscale if xscale else args.xscale)
    ax.set_yscale(yscale if yscale else args.yscale)

    legend = ax.legend(markerscale=3)
    for handle in legend.legend_handles:
        handle.set_alpha(0.85)

    fig.tight_layout()
    filename = f"{measure}{'-'+selected if selected else ''}.{args.fig_format}"
    fig.savefig(f"{args.output_dir}/{filename}")
    plt.close()


def plot_freqs_norms(selected: str = None):
    plot_arrays(
        "Norms of Means (vs. Frequency)",
        "freqs_norms",
        "Token Frequency",
        "Mean Norm",
        xscale="log",
        selected=selected,
    )


def plot_dual_scatters(prefix: str, measure: str):
    for iden in IDENTIFIERS:
        depth, width = iden.split(args.split_char)
        path = f"{args.output_dir}/{prefix}-{iden}.{args.fig_format}"
        proj_m, proj_c = ANALYTICS[iden][measure]
        plot_scatters(
            *(proj_c.cpu(), proj_m.cpu()),
            path,
            f"n_layers={int(depth)}, d={int(width)}",
            dims=args.dims,
            intercept=(0, 0),
        )


if args.eigenvalues:
    plot_arrays("Eigenvalues of Outer Products of Means", "eig_vals", yscale="log")
if args.norms:
    plot_statistic("Norms of Means (CoV)", "norms_cv")
    plot_arrays("Norms of Means (Histogram)", "norms_hist")
    if args.each_model:
        for iden in IDENTIFIERS:
            plot_freqs_norms(iden)
    plot_freqs_norms()
if args.coherence:
    plot_statistic("Coherence (mean)", "coh_mean")
    plot_statistic("Coherence (stddev)", "coh_std")
    plot_statistic("Coherence (CoV)", "coh_cv")
    plot_arrays("Interference of Means", "coh_hist", "Values (Off-Diagonals)")
if args.duality:
    plot_statistic("Self-Duality (Difference)", "dual_diff")
    plot_statistic("Self-Duality (Dot-Product) (Mean)", "dual_dot")
    plot_arrays(
        "Self-Duality (Dot-Product) (Histogram)",
        "dual_hist",
        "Self-Duality (Dot-Product)",
    )
    if args.each_model:
        plot_dual_scatters("dual_dot", "dual_dot_projs")

if args.inv_snr:
    plot_statistic("Inverse Signal-to-Noise Ratio (CoV)", "inv_snr")
    plot_arrays("Inverse Signal-to-Noise Ratio (Histogram)", "inv_snr_hist")

print(LINE_SEP)
