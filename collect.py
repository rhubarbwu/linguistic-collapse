import math
from argparse import ArgumentParser

import torch as pt
from tqdm import tqdm

from lib.collapse import Statistics
from lib.collection import process_batch
from lib.data import load_naive_uint16
from lib.model import get_model, set_model_args

pt.set_grad_enabled(False)

parser = ArgumentParser()
parser.add_argument("-dev", "--device", type=str, default="cpu")
parser.add_argument("-st", "--stage", type=str, default="means")
parser.add_argument("-v", "--verbose", action="store_true")

parser.add_argument("-al", "--align", type=int, default=1024)
parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("-ds", "--dataset_path", type=str, default=".")


parser.add_argument("-o", "--output_dir", type=str, default="stats")
set_model_args(parser)
args = parser.parse_args()


model, C, D = get_model(args)


dataset = load_naive_uint16(args.dataset_path)
N_seqs = len(dataset["tokens"])
N_batches = int(math.ceil(N_seqs / args.batch_size))
extract = lambda i: pt.tensor(dataset["tokens"][str(i)], dtype=pt.int32)


config_str = f"{args.output_dir}/{args.model_size}"

stats = Statistics(C, D, args.device)
if args.stage == "means":  # first pass
    N_seen = stats.load_totals(config_str)
elif args.stage == "covs":  # second pass
    stats.load_totals(config_str)
    N_seen = stats.load_covs_sums(config_str)


N_batches_seen = N_seen // args.batch_size
if N_seen > 0:
    print(f"skipping {N_seen} sequences ({N_batches_seen} batches) already seen...")


for b_idx in tqdm(range(N_batches_seen, N_batches), ncols=79):
    start = b_idx * args.batch_size
    end = min(start + args.batch_size, N_seqs)

    batch = [extract(i) for i in range(start, end)]
    lengths = [len(sample) for sample in batch]
    expected_count = sum(lengths) - pt.tensor(lengths).count_nonzero()
    X, Y = process_batch(model, batch)
    assert X.shape[0] == Y.shape[0] == expected_count

    if args.stage == "means":  # first pass
        stats.collect_means(X, Y, len(batch))
    elif args.stage == "covs":  # second pass
        stats.collect_covs(X, Y, len(batch))

    if (b_idx + 1) % (args.align // args.batch_size) != 0:
        continue  # don't save on most iterations

    if args.stage == "means":
        stats.save_totals(config_str, args.verbose)
    elif args.stage == "covs":
        stats.save_covs_sums(config_str, args.verbose)


dataset.close()


if args.stage == "means":
    stats.save_totals(config_str, args.verbose)
elif args.stage == "covs":
    stats.save_covs_sums(config_str, args.verbose)

idx = stats.counts_in_range(1)
means, mean_G = stats.compute_means(idx)
