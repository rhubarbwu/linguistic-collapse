import math
from dataclasses import dataclass, field
from os import makedirs
from typing import List, Optional, Tuple, Union

import torch as pt
from datasets import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from lib.collapse import Statistics
from lib.model import ModelArguments, strip_model


@dataclass
class CollectArguments:
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": ("Batch size of collection.")},
    )
    device: Optional[str] = field(
        default="cpu",
        metadata={"help": ("Device to run the collection on.")},
    )
    model_ckpt_idx: Optional[int] = field(
        default=9,
        metadata={"help": ("Which checkpoint (by list index) to load for the model.")},
    )
    save_every: Optional[int] = field(
        default=1024,
        metadata={"help": ("How often to save stats.")},
    )
    single: bool = field(
        default=False,
        metadata={"help": ("Save only once.")},
    )
    stage: Optional[str] = field(
        default="means",
        metadata={"help": ("Which measure to collect.")},
    )
    stats_dir: Optional[str] = field(
        default="stats",
        metadata={"help": ("Where to save statistics.")},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": ("Verbose logging and information.")},
    )


def truncate_and_pad(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    assert len(batch) > 0
    if len(batch) == 1:
        return None, batch[0].unsqueeze(0)

    masks = [[1] * len(seq) for seq in batch]
    longest = max([sum(seq) for seq in masks])

    masks = pt.tensor([m + [0] * (longest - len(m)) for m in masks])
    batch = pad_sequence(batch, batch_first=True).clone().detach()

    return masks, batch


def process_batch(
    model: PreTrainedModel,
    batch: List[Tensor],
    stats_device: Union[str, pt.device] = "cpu",
) -> Tuple[Tensor, Tensor]:
    masks, batch = truncate_and_pad(batch)
    output = model(
        batch.to(model.device),
        attention_mask=masks if masks is None else masks.to(model.device),
        output_hidden_states=False,
    )

    embeds = output.logits
    if masks is not None:
        embeds = pt.unsqueeze(masks.to(embeds.device), -1) * embeds

    # offset by one for the next word prediction
    Y = batch[:, 1:].to(stats_device)
    X = embeds[:, :-1].to(stats_device)
    if masks is not None:
        idx = masks[:, 1:].bool()
        Y, X = Y[idx], X[idx]

    return X.squeeze(), Y.squeeze()


def collect_embeddings(
    args: CollectArguments,
    model_args: ModelArguments,
    model: AutoModelForCausalLM,
    data: Dataset,
):
    pt.set_grad_enabled(False)

    means_dir = f"{args.stats_dir}/means@{args.model_ckpt_idx}"
    vars_dir = f"{args.stats_dir}/vars@{args.model_ckpt_idx}"

    makedirs(args.stats_dir, exist_ok=True)
    makedirs(means_dir, exist_ok=True)
    makedirs(vars_dir, exist_ok=True)

    model_name = model_args.model_name_or_path.split("/")[-1]
    means_path = f"{means_dir}/{model_name}@{args.model_ckpt_idx}-means.pt"
    vars_path = f"{vars_dir}/{model_name}@{args.model_ckpt_idx}-vars.pt"

    N_seqs = len(data)
    extract = lambda i: pt.tensor(data[i]["input_ids"], dtype=pt.int32)
    N_batches = int(math.ceil(N_seqs / args.batch_size))

    model, C, D = strip_model(model, args.device)
    stats = Statistics(C, D, args.device)

    N_seen = 0
    if args.stage == "means":  # first pass
        N_seen = stats.load_totals(means_path)
    elif args.stage == "vars":  # second pass
        stats.load_totals(means_path)
        N_seen = stats.load_var_sums(vars_path)
    N_batches_seen = int(math.ceil(N_seen / args.batch_size))
    if N_seen > 0:
        print(f"skipping {N_seen} sequences ({N_batches_seen} batches) already seen...")

    for b_idx in tqdm(range(N_batches_seen, N_batches), ncols=79):
        start = b_idx * args.batch_size
        end = min(start + args.batch_size, N_seqs)
        batch = [extract(i) for i in range(start, end)]
        X, Y = process_batch(model, batch, args.device)
        if args.stage == "means":  # first pass
            stats.collect_means(X, Y, len(batch))
        elif args.stage == "vars":  # second pass
            stats.collect_vars(X, Y, len(batch))

        if (b_idx + 1) % (args.save_every // args.batch_size) != 0:
            continue  # don't save on most iterations

        if args.stage == "means":
            stats.save_totals(means_path, args.verbose)
        elif args.stage == "vars":
            stats.save_var_sums(vars_path, args.verbose)

        if args.single:
            break

    if args.stage == "means":
        stats.save_totals(means_path, args.verbose)
    elif args.stage == "vars":
        stats.save_var_sums(vars_path, args.verbose)
