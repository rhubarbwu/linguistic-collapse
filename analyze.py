from argparse import ArgumentParser
from os.path import exists, isfile

import torch as pt
from pandas import DataFrame

from lib.collapse import Statistics
from lib.model import get_classifier_weights, get_model_stats, split_parts
from lib.statistics import (collect_hist, commit, create_df, triu_mean,
                            triu_std, update_df)
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
parser.add_argument("-o", "--output_file", type=str, default="analysis")
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

args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"W: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"


ANALYSIS = args.model_stats
ANALYSIS |= args.inv_snr | args.duality | args.decisions  # NC1,3,4
ANALYSIS |= args.norms | args.interfere | (args.geodesic is not None)  # (G)NC2


PATHS = {}
for file in sorted(args.input_files, key=lambda x: x.split("/")[-1]):
    if not exists(file) or not isfile(file):
        continue
    iden = identify(file)
    if iden not in PATHS:
        PATHS[iden] = (None, None, None)

    means_path = file if "means" in file else PATHS[iden][0]
    vars_path = file if "vars" in file else PATHS[iden][1]
    decs_path = file if "decs" in file else PATHS[iden][2]
    PATHS[iden] = (means_path, vars_path, decs_path)
    if args.single and None not in PATHS[iden]:
        PATHS = {iden: PATHS[iden]}
        break


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


IDENTIFIERS = sorted(PATHS.keys(), key=lambda x: split_parts(x)[1])  # sort by dim
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


df: DataFrame = create_df(args.output_file)


def triu_stats_histogram(data: pt.Tensor, key: str):
    mean = triu_mean(data)
    std = triu_std(data, mean)

    update_df(df, f"{key}_mean", mean, iden)
    update_df(df, f"{key}_std", std, iden)

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
    if not exists(f"{args.output_file}.h5"):
        commit(f"{args.output_file}", "counts", counts)

    if args.model_stats:
        train_stats = get_model_stats(f"TS{iden}", args)
        for stat_key in train_stats.keys():
            commit(args.output_file, stat_key, train_stats[stat_key], iden)

    if args.inv_snr:  # NC1
        CDNVs = collected.compute_vars(indices)
        if CDNVs is not None and collected.N2 == args.totals[0]:
            mean = triu_mean(CDNVs)
            std = triu_std(CDNVs, mean)
            update_df(df, "cdnv_mean", mean, iden)
            update_df(df, "cdnv_std", std, iden)
            if args.histograms:
                pt.log_(CDNVs)
                bins, edges = collect_hist(CDNVs, args.num_bins, True)
                commit(args.output_file, "cdnv_bins", bins, iden)
                commit(args.output_file, "cdnv_edges", edges, iden)
                del bins, edges

            del CDNVs

    if args.norms:  # NC2
        norms = collected.mean_norms(indices, False, False)
        update_df(df, "norms_mean", norms.mean(), iden)
        update_df(df, "norms_std", norms.std(), iden)

        norms_scaled = collected.mean_norms(indices, False, True)
        update_df(df, "norms_scl_mean", norms_scaled.mean(), iden)
        update_df(df, "norms_scl_std", norms_scaled.std(), iden)

        norms_logged = collected.mean_norms(indices, True, False)
        update_df(df, "norms_log_mean", norms_logged.mean(), iden)
        update_df(df, "norms_log_std", norms_logged.std(), iden)

        norms_logscaled = collected.mean_norms(indices, True, True)
        update_df(df, "norms_logscl_mean", norms_logscaled.mean(), iden)
        update_df(df, "norms_logscl_std", norms_logscaled.std(), iden)

        if args.frequency:
            commit(args.output_file, "norms", norms, iden)
        if args.histograms:
            bins, edges = collect_hist(norms.unsqueeze(0), args.num_bins)
            commit(args.output_file, "norms_bins", bins, iden)
            commit(args.output_file, "norms_edges", edges, iden)
            del bins, edges
        del norms

    if args.interfere:  # NC2
        interfere = collected.interference(indices, args.patch_size)
        triu_stats_histogram(interfere, "interfere")
        del interfere

    if args.geodesic:  # GNC2
        kernel = riesz_kernel if "riesz" in args.geodesic else log_kernel
        distances = collected.geodesic_distances(indices, kernel)
        triu_stats_histogram(distances, f"{args.geodesic}_dist")
        del distances

    if args.duality:  # NC3
        W = get_classifier_weights(f"TinyStories-{iden}", args)

        if W is None:
            print("W: failed to load weights.")
        else:
            dists = collected.dual_dists(W, indices)
            update_df(df, "dists_mean", dists.mean(), iden)
            update_df(df, "dists_std", dists.std(), iden)
            if args.histograms:
                bins, edges = collect_hist(dists, args.num_bins)
                commit(args.output_file, "dists_bins", bins, iden)
                commit(args.output_file, "dists_edges", edges, iden)
                del bins, edges
            del dists

            sims = collected.similarity(W, indices)
            update_df(df, "sims_mean", sims.mean(), iden)
            update_df(df, "sims_std", sims.std(), iden)
            if args.histograms:
                bins, edges = collect_hist(sims, args.num_bins)
                commit(args.output_file, "sims_bins", bins, iden)
                commit(args.output_file, "sims_edges", edges, iden)
                del bins, edges
            del sims

            del W

    if args.decisions:  # NC4
        matches, misses = collected.matches[indices], collected.misses[indices]
        update_df(df, "matches", matches.sum(), iden)
        update_df(df, "misses", misses.sum(), iden)

    if args.force:
        df.to_csv(f"{args.output_file}.csv")

    del collected


df.to_csv(f"{args.output_file}.csv")


print(LINE_SEP)
