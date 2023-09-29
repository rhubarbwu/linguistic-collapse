from argparse import ArgumentParser, Namespace
from typing import Tuple

from torch.nn import Identity
from transformers import AutoModelForCausalLM

from lib.utils import DTYPES, numerate

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


def get_hub_name(name: str, instruct: bool = False):
    if "x" in name:
        parts = name.split("x")
        name = f"{parts[1]}Layer{'s' if int(parts[1]) > 1 else ''}-{parts[0]}"
    if instruct:
        name = f"Instruct-{name}"

    return f"roneneldan/TinyStories-{name}"


sort_by_params = lambda x: numerate(get_model_size(x), "m")
sort_by_dim = lambda x: DIMS[x]


def get_model_size(iden: str) -> Tuple[int, str]:
    size = iden.split("-")[0]
    assert size in DIMS

    return size


def get_model_colour(iden: str) -> str:
    size = iden.split("-")[0]
    if size in COLOURS:
        return COLOURS[size]

    return "#000000"


def set_model_args(parser: ArgumentParser):
    parser.add_argument("-mc", "--model_cache", type=str, default=".")
    parser.add_argument(
        "-ms",
        "--model_size",
        type=str,
        choices=DIMS.keys(),
        default="1M",
    )
    parser.add_argument(
        "-mt",
        "--model_dtype",
        type=str,
        choices=DTYPES.keys(),
        default="fp32",
    )
    parser.add_argument("-bt", "--better", action="store_true")


def get_model(args: Namespace) -> Tuple[AutoModelForCausalLM, int, int]:
    model = AutoModelForCausalLM.from_pretrained(
        get_hub_name(args.model_size),
        cache_dir=args.model_cache,
        torch_dtype=DTYPES[args.model_dtype],
    )
    C, D = model.lm_head.out_features, model.lm_head.in_features
    del model.lm_head

    if args.better:
        model.to_bettertransformer()
    model.lm_head = Identity()
    model.to(args.device)
    return model, C, D
