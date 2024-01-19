from argparse import ArgumentParser
from os.path import exists, isfile

import torch as pt

from lib.collapse import Statistics
from lib.model import get_classifier_weights, get_model_stats, split_parts
from lib.statistics import collect_hist, commit, triu_mean, triu_std
from lib.utils import identify, log_kernel, riesz_kernel

pt.set_grad_enabled(False)

LINE_SEP = "-" * 79
COL_WIDTH = 6


parser = ArgumentParser()

parser.add_argument("-dev", "--device", type=str, default="cpu")
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-1", "--single", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")

parser.add_argument(
    "-total",
    "--totals",
    type=int,
    nargs=3,
    default=(469514249, 229367, 29233),
)

parser.add_argument("-i", "--input_files", type=str, nargs="+", default=[])
parser.add_argument("-o", "--output_file", type=str, default="analysis.h5")
parser.add_argument("-mc", "--model_cache", type=str, default=".")

parser.add_argument("-prog", "--progress", action="store_true")
parser.add_argument("-loss", "--model_stats", action="store_true")
parser.add_argument("-snr", "-cdnv", "--inv_snr", action="store_true")  # NC1 (Galanti)
parser.add_argument("-nor", "--norms", action="store_true")  # (G)NC2
parser.add_argument("-etf", "--interfere", action="store_true")  # NC2
parser.add_argument("-geo", "--geodesic", type=str, default=None)  # GNC2
parser.add_argument("-dual", "--duality", action="store_true")  # NC3
parser.add_argument("-decs", "--decisions", action="store_true")  # NC4
parser.add_argument("-each", "--each_model", action="store_true")
parser.add_argument("-hist", "--histograms", action="store_true")
parser.add_argument("-freq", "--frequency", action="store_true")

parser.add_argument("-mpc", "--min_per_class", type=int, default=1)
parser.add_argument("-Mpc", "--max_per_class", type=int, default=None)
parser.add_argument("-ps", "--patch_size", type=int, default=1024)
parser.add_argument("-nb", "--num_bins", type=int, default=1024)
parser.add_argument("-d", "--dims", type=int, nargs=2, default=None)
parser.add_argument("-split", "--split_char", type=str, default="x")

args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"W: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"


ANALYSIS = args.model_stats
ANALYSIS |= args.inv_snr | args.duality | args.decisions  # NC1,3,4
ANALYSIS |= args.norms | args.interfere | (args.geodesic is not None)  # (G)NC2


PATHS = {}
for file in args.input_files:
    if not exists(file) or not isfile(file):
        continue
    iden = identify(file)
    if iden not in PATHS:
        PATHS[iden] = (None, None, None)

    means_path = file if "means" in file else PATHS[iden][0]
    vars_path = file if "vars" in file else PATHS[iden][1]
    decs_path = file if "decs" in file else PATHS[iden][2]
    PATHS[iden] = (means_path, vars_path, decs_path)


def get_stats(iden):
    stats = Statistics(
        device=args.device,
        load_means=PATHS[iden][0],
        load_vars=PATHS[iden][1],
        load_decs=PATHS[iden][2],
        verbose=False,
    )
    return stats


if args.progress:
    PROGRESS = {}

INCOMPLETE = []


for iden in PATHS:
    collected: Statistics = get_stats(iden)
    Ns = (collected.N1, collected.N2, collected.N3)
    Ns_seqs = (collected.N1_seqs, collected.N2_seqs, collected.N3_seqs)
    if not (Ns[0] == Ns[1] == Ns[2] == args.totals[0]):
        INCOMPLETE.append(iden)

    if args.progress:
        N_unique = collected.counts_in_range(args.min_per_class).shape[0]
        PROGRESS[iden] = (*Ns_seqs, N_unique)
        COL_WIDTH = max(COL_WIDTH, max(len(str(n)) for n in PROGRESS[iden]))

    del collected


IDENTIFIERS = sorted(
    PATHS.keys(), key=lambda x: split_parts(x, args.split_char)[1]
)  # sort by dim
LAST_INDEX = f"total ({len(IDENTIFIERS)})"
LONGEST_IDEN = max(len(LAST_INDEX), max([len(iden) for iden in IDENTIFIERS]))

if args.progress:
    print(LINE_SEP)
    head = [p.rjust(COL_WIDTH) for p in ["means", "vars", "decs", "unique"]]
    print(f"model".ljust(LONGEST_IDEN + 1), *head)
    for iden in IDENTIFIERS:
        Ns = PROGRESS[iden]
        row = [str(n).rjust(COL_WIDTH) for n in Ns]
        print(iden.ljust(LONGEST_IDEN + 1), *row)
    row = [args.totals[1]] * 3 + [args.totals[-1]]
    row = [str(n).rjust(COL_WIDTH) for n in row]
    print(LAST_INDEX.ljust(LONGEST_IDEN + 1), *row)


print(LINE_SEP)
if not ANALYSIS:
    exit()


for iden in INCOMPLETE:
    del PATHS[iden]
    IDENTIFIERS.remove(iden)

if args.single:  # run the first one for debugging purposes
    IDENTIFIERS = IDENTIFIERS[0:1]


def triu_stats_histogram(data: pt.Tensor, key: str):
    mean = triu_mean(data)
    std = triu_std(data, mean)

    commit(args.output_file, f"{key}_mean", mean, iden)
    commit(args.output_file, f"{key}_std", std, iden)

    if args.histograms:
        bins, edges = collect_hist(data, args.num_bins, True)
        commit(args.output_file, f"{key}_bins", bins, iden)
        commit(args.output_file, f"{key}_edges", edges, iden)
        del bins, edges


for iden in IDENTIFIERS:
    if iden not in PATHS:
        print("SKIPPING", iden)
        continue
    if args.verbose:
        print("ANALYZE", iden)

    collected: Statistics = get_stats(iden)

    indices = collected.counts_in_range(args.min_per_class, args.max_per_class)
    counts = collected.counts[indices]
    if "counts" not in file or len(file["counts"]) != len(counts):
        commit(args.output_file, "counts", counts)

    if args.model_stats:
        train_stats = get_model_stats(f"TS{iden}", args)
        for stat_key in train_stats.keys():
            commit(args.output_file, stat_key, train_stats[stat_key], iden)

    if args.inv_snr:  # NC1
        CDNVs = collected.compute_vars(indices)
        if CDNVs is not None and collected.N2 == args.totals[0]:
            mean = triu_mean(CDNVs)
            std = triu_std(CDNVs, mean)
            commit(args.output_file, "cdnv_mean", mean, iden)
            commit(args.output_file, "cdnv_std", std, iden)

            if args.histograms:
                pt.log_(CDNVs)
                bins, edges = collect_hist(CDNVs, args.num_bins, True)
                commit(args.output_file, "cdnv_bins", bins, iden)
                commit(args.output_file, "cdnv_edges", edges, iden)
                del bins, edges

            del CDNVs

    if args.norms:
        norms = collected.mean_norms(indices, False, False)
        commit(args.output_file, "norms_mean", norms.mean(), iden)
        commit(args.output_file, "norms_std", norms.std(), iden)

        norms_scaled = collected.mean_norms(indices, False, True)
        commit(args.output_file, "norms_scaled_mean", norms_scaled.mean(), iden)
        commit(args.output_file, "norms_scaled_std", norms_scaled.std(), iden)

        norms_logged = collected.mean_norms(indices, True, False)
        commit(args.output_file, "norms_logged_mean", norms_logged.mean(), iden)
        commit(args.output_file, "norms_logged_std", norms_logged.std(), iden)

        norms_logscaled = collected.mean_norms(indices, True, True)
        commit(args.output_file, "norms_logscaled_mean", norms_logscaled.mean(), iden)
        commit(args.output_file, "norms_logscaled_std", norms_logscaled.std(), iden)

        if args.frequency:
            commit(args.output_file, "norms", norms, iden)
        if args.histograms:
            bins, edges = collect_hist(norms.unsqueeze(0), args.num_bins)
            commit(args.output_file, "norms_bins", bins, iden)
            commit(args.output_file, "norms_edges", edges, iden)
            del bins, edges
        del norms

    if args.interfere:
        interfere = collected.interference(indices, args.patch_size)
        triu_stats_histogram(interfere, "interfere")
        del interfere

    if args.geodesic:
        kernel = riesz_kernel if "riesz" in args.geodesic else log_kernel
        distances = collected.geodesic_distances(indices, kernel)
        triu_stats_histogram(distances, f"geodesic_{args.geodesic}")
        del distances

    if args.duality:
        W = get_classifier_weights(f"TS{iden}", args)
        if W is not None:
            dual_diff = collected.diff_duality(W, indices)
            commit(args.output_file, "dual_diff", dual_diff, iden)
            del dual_diff

            dual_dot = collected.dot_duality(W, indices)
            commit(args.output_file, "dual_dot_mean", dual_dot.mean(), iden)
            commit(args.output_file, "dual_dot_std", dual_dot.std(), iden)
            del W

            if args.histograms:
                bins, edges = collect_hist(dual_dot, args.num_bins)
                commit(args.output_file, "dual_bins", bins, iden)
                commit(args.output_file, "dual_edges", edges, iden)
                del bins, edges

            del dual_dot

    if args.decisions:
        matches, misses = collected.matches[indices], collected.misses[indices]
        commit(args.output_file, "matches", matches.sum(), iden)
        commit(args.output_file, "misses", misses.sum(), iden)

        matches = matches.float()
        commit(args.output_file, "matches_mean", matches.mean(), iden)
        commit(args.output_file, "matches_std", matches.std(), iden)
    del collected


print(LINE_SEP)
