from argparse import ArgumentParser
from os.path import exists, isfile

import torch as pt
from pandas import DataFrame

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from lib.collapse import CollapseStatistics
from lib.model import get_classifier_weights, get_model_stats, split_parts
from lib.statistics import commit, create_df, save_metrics, update_df
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

parser.add_argument("-mpc", "--min_per_class", type=int, default=1)
parser.add_argument("-Mpc", "--max_per_class", type=int, default=None)
parser.add_argument("-ts", "--tile_size", type=int, default=1024)

args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"WARN: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"


REQ_MEANS = args.inv_snr | args.norms | args.interfere | args.duality
REQ_MEANS |= args.kernel is not None
ANALYSIS = args.model_stats | REQ_MEANS | args.decisions
mpc, Mpc = args.min_per_class, args.max_per_class


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


def get_stats(iden: str, force_cpu: bool = False):
    stats = CollapseStatistics(
        device="cpu" if force_cpu else args.device,
        means_path=PATHS[iden][0],
        vars_path=PATHS[iden][1],
        decs_path=PATHS[iden][2],
        verbose=False,
    )
    return stats


if args.progress:
    PROGRESS = {}


for iden in tqdm(PATHS):
    nc_stats: CollapseStatistics = get_stats(iden, True)

    if args.progress:
        N_unique = nc_stats.means_accum.filter_indices_by_n_samples(mpc, Mpc).shape[0]
        Ns_seqs = (nc_stats.N_seqs_means, nc_stats.N_seqs_vars, nc_stats.N_seqs_decs)
        PROGRESS[iden] = (*Ns_seqs, N_unique)
        COL_WIDTH = max(COL_WIDTH, max(len(str(n)) for n in PROGRESS[iden]))

    del nc_stats


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
                                     similarities, simplex_etf_error,
                                     variability_cdnv)

for iden in tqdm(IDENTIFIERS):
    if iden not in PATHS:
        print("SKIPPING", iden)
        continue
    if args.verbose:
        print("ANALYZE", iden)

    nc_stats: CollapseStatistics = get_stats(iden)
    indices = nc_stats.means_accum.filter_indices_by_n_samples(mpc, Mpc)
    if not exists(f"{args.output_file}.h5"):
        commit(f"{args.output_file}", "Ns_toks", nc_stats.means_accum.ns_samples)

    if args.model_stats and missing("n_params", iden):
        try:
            train_stats = get_model_stats(f"TinyStories-{iden}", args)
            for key in train_stats.keys():
                update_df(df, key, train_stats[key], iden)
        except:
            print(f"WARN: failed to load info for {iden}.")

    if REQ_MEANS:
        M, mG = nc_stats.means_accum.compute(indices)

    if args.inv_snr and missing("cdnv_var", iden):  # NC1
        N_vars, N_means = nc_stats.N_seqs_vars, nc_stats.N_seqs_means
        if N_vars < N_means:
            print(f"W: vars for {iden} incomplete ({N_vars} < {N_means}); skipping")
        elif N_vars > N_means:
            print(f"E: too many vars ({N_vars} > {N_means})! skipping")
        else:
            V = nc_stats.vars_accum.compute(indices)[0]
            update_df(df, "cdnv", variability_cdnv(V, M, 2, args.tile_size), iden)
            del V

    if args.norms and missing("norms_var", iden):  # NC2 equinorm
        norms = mean_norms(M, mG)
        save_metrics(df, norms, "norms", iden)
        del norms
    if args.norms and missing("norms_log_var", iden):  # NC2 equinorm (log)
        norms = mean_norms(M, mG, [pt.log])
        save_metrics(df, norms, "norms_log", iden)
        del norms

    if args.interfere and missing("interfere_var", iden):  # NC2 simplex ETF
        interference = interference_grid(M, mG)
        update_df(df, "etf_error", simplex_etf_error(M, mG), iden)
        save_metrics(df, interference, "interfere", iden, True)
        del interference

    if args.kernel and missing(f"{args.kernel}_kern_var", iden):  # GNC2
        kernel = riesz_kernel if "riesz" in args.kernel else log_kernel
        dists = kernel_grid(M, mG, kernel, args.tile_size)
        save_metrics(df, dists, f"{args.kernel}_kern", iden, True)
        del dists

    if args.duality and missing("dual_error", iden):  # NC3 duality
        W = get_classifier_weights(f"TinyStories-{iden}", args)
        if W is None:
            print(f"WARN: failed to load weights for {iden}.")
        else:
            W = W if indices is None else W[indices]
            dual_error = self_duality_error(W, M.to(W.dtype), mG.to(W.dtype))
            update_df(df, "dual_error", dual_error, iden)
            save_metrics(df, similarities(W, M, mG), "simdot", iden)
            save_metrics(df, similarities(W, M, mG, True), "simcos", iden)
            save_metrics(df, distance_norms(W, M, mG), "dists", iden)
        del W

    if args.decisions and missing("hits", iden):  # NC4 agreement
        hits = nc_stats.decs_accum.totals[indices]
        misses = nc_stats.decs_accum.ns_samples[indices] - hits
        update_df(df, "hits", hits.sum(), iden)
        update_df(df, "misses", misses.sum(), iden)

    del nc_stats
    df.to_csv(f"{args.output_file}.csv")

print(LINE_SEP)
