import os
from argparse import ArgumentParser, Namespace
from typing import Tuple

import matplotlib.pyplot as plt
import torch as pt
from torch import Tensor
from torch.nn import Identity
from transformers import AutoModelForCausalLM

from lib.statistics import collect_hist
from lib.utils import DTYPES, inner_product, normalize, numerate
from lib.visualization import plot_histogram, set_vis_args

DIMS = {
    "1Mx8": 64,
    "3Mx8": 128,
    "8Mx8": 256,
    "28Mx8": 512,
    "33Mx4": 768,
    "21Mx1": 1024,
    "33Mx2": 1024,
}

COLOURS = {
    "1Mx8": "#1f77b4",
    "3Mx8": "#ff7f0e",
    "8Mx8": "#2ca02c",
    "28Mx8": "#d62728",
    "33Mx4": "#9467bd",
    "21Mx1": "#8c564b",
    "33Mx2": "#e377c2",
    # "12b": "#7f7f7f",
}

HUB_NAMES = {
    "1Mx8": "1M",
    "3Mx8": "3M",
    "8Mx8": "8M",
    "28Mx8": "28M",
    "33Mx4": "33M",
    "21Mx1": "1Layer-21M",
    "33Mx2": "2Layers-33M",
}


def set_model_args(parser: ArgumentParser):
    parser.add_argument("-mc", "--model_cache", type=str, default=".")
    parser.add_argument(
        "-ms",
        "--model_size",
        type=str,
        choices=DIMS.keys(),
        default="1Mx8",
    )
    parser.add_argument(
        "-mt",
        "--model_dtype",
        type=str,
        choices=DTYPES.keys(),
        default="fp32",
    )
    parser.add_argument(
        "-mss",
        "--model_sizes",
        type=str,
        nargs="+",
        choices=DIMS.keys(),
        default=DIMS.keys(),
    )
    parser.add_argument("-bt", "--better", action="store_true")
    parser.add_argument("-f", "--force_load", action="store_true")


def get_hub_name(name: str, instruct: bool = False):
    if "x" in name:
        parts = name.split("x")
        name = f"{parts[1]}Layer{'s' if int(parts[1]) > 1 else ''}-{parts[0]}"
    if instruct:
        name = f"Instruct-{name}"

    return f"roneneldan/TinyStories-{name}"


sort_by_params = lambda x: numerate(x, "m")
sort_by_dim = lambda x: DIMS[x]


def get_model_colour(iden: str) -> str:
    iden = iden.split("-")[0]
    if iden in COLOURS:
        return COLOURS[iden]

    return "#000000"


def get_classifier_weights(args: Namespace) -> Tensor:
    classifier_file = f"{args.model_cache}/{args.model_size}-classifier.pt"
    if not os.path.exists(classifier_file):
        print(f"classifier weights file for {args.model_size} not found...")

        if not args.force_load:
            if not input("load? ").upper().startswith("Y"):
                return None

        get_model(args)

    return pt.load(classifier_file, args.device)


def get_model(args: Namespace) -> Tuple[AutoModelForCausalLM, int, int]:
    model = AutoModelForCausalLM.from_pretrained(
        f"roneneldan/TinyStories-{HUB_NAMES[args.model_size]}",
        cache_dir=args.model_cache,
        torch_dtype=DTYPES[args.model_dtype],
    )
    C, D = model.lm_head.out_features, model.lm_head.in_features
    W = list(model.lm_head.parameters())[0].detach()
    del model.lm_head
    model.lm_head = Identity()

    if args.better:
        model.to_bettertransformer()

    classifier_file = f"{args.model_cache}/{args.model_size}-classifier.pt"
    if not os.path.exists(classifier_file):
        print(f"caching weights for {args.model_size} in {classifier_file}", flush=True)
        pt.save(W, classifier_file)

    model.to(args.device)
    return model, C, D


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dev", "--device", type=str, default="cpu")
    parser.add_argument("-ps", "--patch_size", type=int, default=1024)
    parser.add_argument("-idx", "--indices", type=str, default=None)
    set_model_args(parser)
    set_vis_args(parser)
    args = parser.parse_args()

    hist_keys = ["inner", "norms"]
    hists = {k: plt.subplots(figsize=(args.fig_size)) for k in hist_keys}

    indices = None if not args.indices else pt.load(args.indices, args.device)

    for key, (fig, ax) in hists.items():
        key_str = "Norms" if key == "norms" else "Inner Product"
        ax.set_title(
            f"{key_str} of TinyStories Classifier Weights (Subset of GPT Neo's Vocabulary)"
        )
        ax.set_xlabel(f"value ({args.num_bins} bins)")
        ax.set_ylabel("frequency")
        if key == "norms":
            ax.set_yscale("log")

    for iden in args.model_sizes:
        args.model_size = iden
        W = get_classifier_weights(args)
        W = W if indices is None else W[indices]
        if W is None:
            continue

        label, color = f"{iden} ({DIMS[iden]})", get_model_colour(iden)

        norms_ax = hists["norms"][1]
        W_norms = W.norm(dim=-1).unsqueeze(0)
        hist, edges = collect_hist(W_norms, args.num_bins, desc="norms")
        plot_histogram(norms_ax, hist, edges, label, color)

        W_normed = normalize(W - W.mean(dim=0, keepdim=True))
        inner_prod = inner_product(W_normed, patch_size=args.patch_size)
        inner_ax = hists["inner"][1]
        inner_ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        hist, edges = collect_hist(inner_prod, args.num_bins, True, desc="inner prod")
        plot_histogram(inner_ax, hist, edges, label, color)

        del W, norms_ax, W_norms, hist, edges, W_normed, inner_prod

    for key, (fig, ax) in hists.items():
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{args.output_dir}/cls_{key}.{args.fig_format}")

    # get_model(args)
