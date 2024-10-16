from argparse import ArgumentParser
from os.path import exists, isfile

import torch as pt
from pandas import DataFrame

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from lib.collapse import Statistics
from lib.model import get_classifier_weights, get_model_stats, split_parts
from lib.statistics import commit, create_df, save_stats, update_df
from lib.utils import identify, is_float

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
    default=(469514249, 229367, 29233),  # TinyStories train set, 2 workers
)

parser.add_argument("-i", "--input_files", type=str, nargs="+", default=[])
parser.add_argument("-o", "--output_file", type=str, default="analysis")
parser.add_argument("-mc", "--model_cache", type=str, default=".")

parser.add_argument("-prog", "--progress", action="store_true")
parser.add_argument("-loss", "--model_stats", action="store_true")
parser.add_argument("-snr", "-cdnv", "--inv_snr", action="store_true")  # NC1 (Galanti)
parser.add_argument("-nor", "--norms", action="store_true")  # (G)NC2
parser.add_argument("-etf", "--interfere", action="store_true")  # NC2
parser.add_argument("-kern", "--kernel", type=str, default=None)  # GNC2
parser.add_argument("-dual", "--duality", action="store_true")  # NC3
parser.add_argument("-decs", "--decisions", action="store_true")  # NC4
parser.add_argument("-each", "--each_model", action="store_true")
parser.add_argument("-freq", "--frequency", action="store_true")

parser.add_argument("-mpc", "--min_per_class", type=int, default=1)
parser.add_argument("-Mpc", "--max_per_class", type=int, default=None)
parser.add_argument("-ts", "--tile_size", type=int, default=1024)

args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"WARN: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"


REQ_MEANS = args.norms | args.interfere | (args.kernel is not None) | args.duality
ANALYSIS = args.model_stats | args.inv_snr | REQ_MEANS | args.decisions


PATHS = {}
for file in sorted(args.input_files, key=lambda x: x.split("/")[-1]):
    if not exists(file) or not isfile(file):
        continue
    iden = identify(file)
    paths = [None, None, None] if iden not in PATHS else PATHS[iden]

    if "means" in file:
        paths[0] = file
    elif "vars" in file:
        paths[1] = file
    elif "decs" in file:
        paths[2] = file
    else:
        continue

    PATHS[iden] = paths
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


for iden in tqdm(PATHS):
    collected: Statistics = get_stats(iden)
    Ns = (collected.N1, collected.N2, collected.N3)
    Ns_seqs = (collected.N1_seqs, collected.N2_seqs, collected.N3_seqs)
    if not (Ns[0] == Ns[1] == args.totals[0]):
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

if args.single:
    print("WARN: only analyzing one model for debugging purposes")
    IDENTIFIERS = IDENTIFIERS[0:1]


df: DataFrame = create_df(args.output_file)
missing = lambda k, i: not (k in df and i in df.index and is_float(df[k][i]))
if args.force:
    missing = lambda k, i: True

from neural_collapse.kernels import kernel_grid, log_kernel, riesz_kernel
from neural_collapse.measure import (distance_norms, interference_grid,
                                     mean_norms, self_duality_error,
                                     similarities, simplex_etf_error)

for iden in tqdm(IDENTIFIERS):
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
        try:
            train_stats = get_model_stats(f"TinyStories-{iden}", args)
            for key in train_stats.keys():
                update_df(df, key, train_stats[key], iden)
        except:
            print(f"WARN: failed to load info for {iden}.")

    if REQ_MEANS:
        M, mG = collected.compute_means(indices)

    if args.inv_snr and missing("cdnv_var", iden):  # NC1
        CDNVs = collected.compute_vars(indices, args.tile_size)
        if CDNVs is not None and collected.N2 == args.totals[0]:
            save_stats(df, CDNVs, "cdnv", iden, True)
            del CDNVs

    if args.norms and missing("norms_var", iden):  # NC2 equinorm
        norms = mean_norms(M, mG)
        save_stats(df, norms, "norms", iden)
        if args.frequency:
            commit(args.output_file, "norms", norms, iden)
        del norms

    if args.interfere and missing("interfere_var", iden):  # NC2 simplex ETF
        interference = interference_grid(M, mG)
        update_df(df, "etf_error", simplex_etf_error(M, mG), iden)
        save_stats(df, interference, "interfere", iden, True)
        del interference

    if args.kernel and missing(f"{args.kernel}_kern_var", iden):  # GNC2
        kernel = riesz_kernel if "riesz" in args.kernel else log_kernel
        dists = kernel_grid(M, mG, kernel, args.tile_size)
        save_stats(df, dists, f"{args.kernel}_kern", iden, True)
        del dists

    if args.duality and missing("dual_error", iden):  # NC3 duality
        W = get_classifier_weights(f"TinyStories-{iden}", args)
        if W is None:
            print(f"WARN: failed to load weights for {iden}.")
        else:
            W = W if indices is None else W[indices]
            dual_error = self_duality_error(M.to(W.dtype), W, mG.to(W.dtype))
            update_df(df, "dual_error", dual_error, iden)
            save_stats(df, similarities(M, W, mG), "simdot", iden)
            save_stats(df, similarities(M, W, mG, True), "simcos", iden)
            save_stats(df, distance_norms(W, M, mG), "dists", iden)
        del W

    if args.decisions and missing("hits", iden):  # NC4 agreement
        hits, misses = collected.matches[indices], collected.misses[indices]
        update_df(df, "hits", int(hits.sum()), iden)
        update_df(df, "misses", int(misses.sum()), iden)

    del collected
    df.to_csv(f"{args.output_file}.csv")

print(LINE_SEP)
