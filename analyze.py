from argparse import ArgumentParser
from os.path import exists, isfile
from random import sample

import torch as pt
from h5py import File

from lib.collapse import Statistics
from lib.model import get_classifier_weights, get_model_loss, split_parts
from lib.statistics import ANALYSIS_FILE, collect_hist, meanify_diag, replace
from lib.utils import identify, inner_product
from lib.visualization import TOO_BIG, visualize_matrix

pt.set_grad_enabled(False)

LINE_SEP = "-" * 79
COL_WIDTH = 6


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
parser.add_argument("-o", "--output_file", type=str, default=ANALYSIS_FILE)
parser.add_argument("-mc", "--model_cache", type=str, default=".")
parser.add_argument("-f", "--force_load", action="store_true")

parser.add_argument("-prog", "--progress", action="store_true")
parser.add_argument("-loss", "--train_loss", action="store_true")
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
parser.add_argument("-nb", "--num_bins", type=int, default=1024)
parser.add_argument("-d", "--dims", type=int, nargs=2, default=None)
parser.add_argument("-split", "--split_char", type=str, default="x")

args = parser.parse_args()

if "cuda" in args.device and not pt.cuda.is_available():
    print(f"W: CUDA device {args.device} unavailable; defaulting to CPU")
    args.device = "cpu"


if args.analysis:
    args.eigenvalues = args.norms = args.coherence = True
    args.train_loss = args.duality = args.inv_snr = True
args.analysis |= args.eigenvalues | args.norms | args.coherence
args.analysis |= args.train_loss | args.duality | args.inv_snr


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


def get_stats(iden):
    stats = Statistics(
        device=args.device,
        load_means=PATHS[iden][0],
        load_vars=PATHS[iden][1],
        verbose=False,
    )
    return stats


if args.progress:
    PROGRESS = {}

INCOMPLETE = []


for iden in PATHS:
    collected: Statistics = get_stats(iden)
    Ns = (collected.N1, collected.N1_seqs, collected.N2, collected.N2_seqs)
    if Ns[0] != args.totals[0] or Ns[1] != args.totals[1]:
        INCOMPLETE.append(iden)

    if args.progress:
        PROGRESS[iden] = (*Ns, collected.counts_in_range(args.min_per_class).shape[0])
        COL_WIDTH = max(COL_WIDTH, max(len(str(n)) for n in PROGRESS[iden]))

    del collected


IDENTIFIERS = sorted(PATHS.keys(), key=lambda x: split_parts(x)[2])
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
    del PATHS[iden]
    IDENTIFIERS.remove(iden)

file = File(args.output_file, "a")


for iden in IDENTIFIERS:
    if iden not in PATHS:
        continue

    collected: Statistics = get_stats(iden)

    indices = collected.counts_in_range(args.min_per_class, args.max_per_class)
    counts = collected.counts[indices]
    if "counts" not in file or len(file["counts"]) != len(counts):
        replace(file, "counts", counts)

    means, mean_G = collected.compute_means(indices)
    if args.dims is None:
        args.dims = sample(list(range(len(indices))), 2)

    if args.train_loss:
        loss = get_model_loss(f"TS{iden}", args)
        replace(file, "loss", loss, iden)
        print(iden, loss)

    if args.eigenvalues:
        outer = inner_product(means.T, args.patch_size, "eigs outer")
        if outer.shape[0] > TOO_BIG:
            print(f"W: {outer.shape[0]} > {TOO_BIG}, eigenvalues disabled")
        else:
            eig_vals = pt.linalg.eigvalsh(outer)
            replace(file, "eig_vals", eig_vals.real, iden)

    if args.norms:
        norms = means.norm(dim=-1) ** 2
        replace(file, "norms", norms, iden)
        replace(file, "norms_mean", norms.mean(), iden)
        replace(file, "norms_std", norms.std(), iden)

        bins, edges = collect_hist(norms.unsqueeze(0), args.num_bins, desc="norms hist")
        replace(file, "norms_bins", bins, iden)
        replace(file, "norms_edges", edges, iden)
        del norms, bins, edges

    if args.coherence:
        inner = collected.coherence(indices, args.patch_size)
        replace(file, "coh_mean", inner.mean(), iden)
        replace(file, "coh_std", inner.std(), iden)

        bins, edges = collect_hist(inner, args.num_bins, True, desc="coh hist")
        replace(file, "coh_bins", bins, iden)
        replace(file, "coh_edges", edges, iden)

        del inner, bins, edges

    if args.duality:
        W = get_classifier_weights(f"TS{iden}", args)
        if W is not None:
            dual_diff = collected.diff_duality(W, indices)
            replace(file, "dual_diff", dual_diff, iden)

            dual_dot, proj_m, proj_c = collected.dot_duality(W, indices, args.dims)
            replace(file, "dual_dot_mean", dual_dot.mean(), iden)
            replace(file, "dual_dot_std", dual_dot.std(), iden)
            del W, proj_m, proj_c

            bins, edges = collect_hist(dual_dot, args.num_bins, desc="dual hist")
            replace(file, "dual_bins", bins, iden)
            replace(file, "dual_edges", edges, iden)
            del dual_dot, bins, edges

    if args.inv_snr:
        CDNVs = collected.compute_vars(indices)
        if CDNVs is not None and collected.N2 == args.totals[0]:
            meanify_diag(CDNVs)
            CDNVs = pt.log(CDNVs)
            replace(file, "cdnv_mean", CDNVs.mean(), iden)
            replace(file, "cdnv_std", CDNVs.std(), iden)
            if args.each_model:
                matpath = f"{args.output_dir}/cdnv-{iden}.{args.fig_format}"
                visualize_matrix(CDNVs, matpath)

            bins, edges = collect_hist(CDNVs, args.num_bins, True)
            replace(file, "cdnv_bins", bins, iden)
            replace(file, "cdnv_edges", edges, iden)
            del CDNVs, bins, edges

    del collected


file.close()
print(LINE_SEP)
