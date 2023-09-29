from argparse import ArgumentParser
from os import makedirs

import torch

from lib.collapse import Statistics
from lib.model import DIMS, get_model_colour, sort_by_dim, sort_by_params
from lib.statistics import collect_hist
from lib.utils import identify, inner_product
from lib.visualization import TOO_BIG, plot_graph, plot_histogram

LINE_SEP = "-" * 79

parser = ArgumentParser()
parser.add_argument("-dev", "--device", type=str, default="cpu")
parser.add_argument("-ds", "--dataset_path", type=str, default=".")
parser.add_argument(
    "-total",
    "--totals",
    type=int,
    nargs=2,
    default=(469753028, 2119719),
)

parser.add_argument("-i", "--input_files", type=str, nargs="+", default=[])
parser.add_argument("-o", "--output_dir", type=str, default="figures")
parser.add_argument("-mc", "--model_cache", type=str, default=".")

parser.add_argument("-prog", "--progress", action="store_true")
parser.add_argument("-eig", "--eigenvalues", action="store_true")
parser.add_argument("-nor", "--norms", action="store_true")
parser.add_argument("-coh", "--coherence", action="store_true")
parser.add_argument("-dual", "--duality", action="store_true")
parser.add_argument("-snr", "--inv_snr", action="store_true")
parser.add_argument("-all", "--analysis", action="store_true")


parser.add_argument("-k", "--n_clusters", type=int, default=1)
parser.add_argument("-mpc", "--min_per_class", type=int, default=1)
parser.add_argument("-Mpc", "--max_per_class", type=int, default=None)
parser.add_argument("-ps", "--patch_size", type=int, default=1024)


parser.add_argument("-fmt", "--fig_format", type=str, default="png")
parser.add_argument("-fs", "--fig_size", type=int, nargs=2, default=(9, 6))
parser.add_argument("-mag", "--magnitude", type=str, default="m")
parser.add_argument("-nb", "--num_bins", type=int, default=1024)
parser.add_argument("-sort", "--sort_by", type=str, default="params")
parser.add_argument("-xs", "--xscale", type=str, default="linear")
parser.add_argument("-ys", "--yscale", type=str, default="linear")
args = parser.parse_args()


if args.analysis:
    args.eigenvalues = args.norms = args.coherence = args.inv_snr = args.duality = True
args.analysis |= args.eigenvalues | args.norms | args.coherence
args.analysis |= args.duality | args.inv_snr


model_sorter = sort_by_dim if "dim" in args.sort_by else sort_by_params

makedirs(args.output_dir, exist_ok=True)


PATHS = {}
for file in args.input_files:
    iden = identify(file)
    if iden not in PATHS:
        PATHS[iden] = (None, None)

    means_path = file if "means" in file else PATHS[iden][0]
    covs_path = file if "covs" in file else PATHS[iden][1]
    PATHS[iden] = (means_path, covs_path)

longest_iden = max(5, max([len(iden) for iden in PATHS]))


if args.progress:
    COL_WIDTH = 6
    PROGRESS = {}
if args.analysis:
    STATISTICS = {}

for iden, (means_path, covs_path) in PATHS.items():
    collected = Statistics(load_means=means_path, load_covs=covs_path, verbose=False)
    Ns = (collected.N1, collected.N1_seqs, collected.N2, collected.N2_seqs)

    if args.analysis:
        STATISTICS[iden] = collected

    if args.progress:
        PROGRESS[iden] = (*Ns, collected.counts_in_range(args.min_per_class).shape[0])
        COL_WIDTH = max(COL_WIDTH, max(len(str(n)) for n in PROGRESS[iden]))

    del collected

if args.progress:
    print(LINE_SEP)
    head = [p.rjust(COL_WIDTH) for p in ["means", "(seqs)", "covs", "(seqs)", "unique"]]
    print("".ljust(longest_iden + 1), *head)
    for iden in sorted(DIMS, key=model_sorter):
        Ns = PROGRESS[iden]
        row = [str(n).rjust(COL_WIDTH) for n in Ns]
        print(iden.ljust(longest_iden + 1), *row)
    row = [str(n).rjust(COL_WIDTH) for n in 2 * args.totals]
    print("total".ljust(longest_iden + 1), *row)


print(LINE_SEP)
if not args.analysis:
    exit()


ANALYTICS = {}
for iden in sorted(DIMS, key=model_sorter):
    if iden not in STATISTICS:
        continue
    collected: Statistics = STATISTICS[iden]
    indices = collected.counts_in_range(args.min_per_class, args.max_per_class)
    counts = collected.counts[indices]
    means, mean_G = collected.compute_means(indices)

    analytics = {}

    if args.eigenvalues:
        outer = inner_product(means.T, args.patch_size, "eigs outer")
        if outer.shape[0] > TOO_BIG:
            print(f"W: {outer.shape[0]} > {TOO_BIG}, eigenvalues disabled")
        else:
            eig_vals = torch.linalg.eigvalsh(outer)
            analytics["eig_vals"] = eig_vals.real

    if args.norms:
        norms = means.norm(dim=-1) ** 2
        norms_hist = collect_hist(norms.unsqueeze(0), args.num_bins, desc="norms hist")
        analytics["norms_cv"] = norms.std() / norms.mean()
        analytics["norms_hist"] = norms_hist
        analytics["freqs_norms"] = torch.stack((counts, norms))
        del norms

    if args.coherence:
        inner = collected.coherence(indices, args.n_clusters, args.patch_size)
        hist, edges = collect_hist(inner, args.num_bins, True, desc="coh hist")
        triu_vals = inner[torch.ones_like(inner).triu(1) != 0]
        assert len(triu_vals) == hist.sum().int().item()
        del inner

        analytics["coh_hist"] = (hist, edges)
        analytics["coh_mean"] = triu_vals.mean().cpu()
        analytics["coh_std"] = triu_vals.std().cpu()
        del triu_vals

    if analytics:
        ANALYTICS[iden] = analytics


range_str = ""
if args.min_per_class:
    if args.max_per_class:
        range_str = f" ({args.min_per_class} \u2264 counts < {args.max_per_class})"
    else:
        range_str = f" (counts \u2265 {args.min_per_class})"

import matplotlib.pyplot as plt

from lib.utils import MAG_STRS, MAGS, numerate


def plot_statistic(measure: str, key: str):
    fig, ax = plt.subplots(figsize=args.fig_size)

    idens = sorted(ANALYTICS.keys(), key=model_sorter)
    if "dim" in args.sort_by:
        nums = [DIMS[iden] for iden in idens]
    else:
        nums = [numerate(iden, "m") for iden in idens]

    vals = [ANALYTICS[iden].get(key, None) for iden in idens]

    ax.scatter(nums, vals, marker="o", label="TinyStories")  # plot of scatter
    for x, y, label in zip(nums, vals, idens):
        ax.annotate(label, (x, y))
    ax.legend()
    ax.set_ylabel(measure.lower())
    if "dim" in args.sort_by:
        ax.set_title(f"{measure} vs. Model Size{range_str}")
        ax.set_xlabel(f"Hidden Dimension")
    else:
        ax.set_title(f"{measure} vs. Model Dimension{range_str}")
        ax.set_xlabel(f"Parameter Count ({MAG_STRS[MAGS.index(args.magnitude)]}s)")

    fig.tight_layout()
    fig.savefig(f"{args.output_dir}/{key}.{args.fig_format}")


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

    identifiers = sorted(ANALYTICS.keys(), key=model_sorter)
    if selected:
        identifiers = [selected]

    for iden in identifiers:
        array = ANALYTICS[iden][measure]
        if array is None:
            continue

        color = get_model_colour(iden)
        if "hist" in measure:
            plot_histogram(ax, *array, f"{iden} ({DIMS[iden]})", color)
            continue

        if len(array.shape) == 1:
            intercept = (len(array) - 1, array[-1])
            plot_graph(ax, array, f"{iden} ({DIMS[iden]})", color, intercept)
        else:
            ax.scatter(*array, 5, color, alpha=0.15, label=f"{iden} ({DIMS[iden]})")

    if "hist" in measure:
        xlabel, ylabel = "Value", "Frequency"
    elif "eig" in measure:
        xlabel, ylabel = "Dimension", "Eigenvalue"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} {'('+iden+') ' if iden else ''}{range_str}")
    ax.set_xscale(xscale if xscale else args.xscale)
    ax.set_yscale(yscale if yscale else args.yscale)
    if not selected:
        ax.legend()

    fig.tight_layout()
    filename = f"{measure}{'-'+selected if selected else ''}.{args.fig_format}"
    fig.savefig(f"{args.output_dir}/{filename}")


def plot_freqs_norms(selected: str = None):
    plot_arrays(
        "Norms of Means (Frequency vs. Norm)",
        "freqs_norms",
        "Token Frequency",
        "Mean Norm",
        xscale="log",
        selected=selected,
    )


if args.eigenvalues:
    plot_arrays("Eigenvalues of Outer Products of Means", "eig_vals", yscale="log")
if args.norms:
    plot_statistic("Norms of Means (stddev/mean)", "norms_cv")
    plot_arrays("Norms of Means (Histogram)", "norms_hist")

    for iden in sorted(ANALYTICS.keys(), key=model_sorter):
        plot_freqs_norms(iden)
    plot_freqs_norms()

if args.coherence:
    plot_statistic("Coherence (mean)", "coh_mean")
    plot_statistic("Coherence (stddev)", "coh_std")
    plot_arrays("Interference of Means", "coh_hist", "Values (Off-Diagonals)")


print(LINE_SEP)
