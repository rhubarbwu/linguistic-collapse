import gc
import os
from os.path import isfile
from typing import Any, List, Set, Tuple

import numpy as np
import torch
import torch.linalg as la
from matplotlib.axes import Axes
from torch import (Tensor, bfloat16, float16, float32, int8, int16, int32,
                   int64, uint8)
from tqdm import tqdm

normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + torch.finfo(x.dtype).eps)
get_dev_name = lambda d: "cpu" if d == "cpu" else torch.cuda.get_device_name(d)


DTYPES = {
    "fp32": float32,
    "fp16": float16,
    "bf16": bfloat16,
}


def select_int_type(value: int = 0) -> torch.dtype:
    if value is None:
        return None
    for dtype in [int8, uint8, int16, int32, int64]:
        if value < torch.iinfo(dtype).max:
            return dtype

    raise ValueError(f"{value} too big")


MAGS = ["q", "t", "b", "m", "k"]
MAG_STRS = ["quadrillion", "trillion", "billion", "million", "thousand"]


def numerate(size: str, ref: str = "k") -> float:
    ref_idx = MAGS.index(ref)
    size = size.split("x")[0].lower()

    for i, scale in enumerate(MAGS[:-1]):
        if scale in size:
            num = float(size.replace(scale, ""))
            scaled = num * 1000 ** (ref_idx - i)
            return scaled


def clean_up(*garbage: List[Any]):
    for g in garbage:
        del g
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CRUFT = [".pt", "means-", "covs-", "-means", "-covs"]


def identify(path: str) -> str:
    identifier = path.split("/")[-1]
    cruft_matches = 0
    for cruft in CRUFT:
        if cruft in identifier:
            cruft_matches += 1
        identifier = identifier.replace(cruft, "")
    return identifier if cruft_matches >= 2 else None


def pathify(identifier: str, stats_dir: str) -> Tuple[str, str]:
    means_path = f"{stats_dir}/means-{identifier}.pt"
    if not isfile(means_path):
        means_path = f"{stats_dir}/{identifier}-means.pt"
        if not isfile(means_path):
            return None, None
    means_path = means_path.replace("//", "/")

    covs_path = f"{stats_dir}/covs-{identifier}.pt"
    if not isfile(covs_path):
        covs_path = f"{stats_dir}/{identifier}-covs.pt"
        if not isfile(covs_path):
            covs_path = None

    covs_path = None if covs_path is None else covs_path.replace("//", "/")

    return means_path, covs_path


def scrub(string: str, antis: List[str]):
    for anti in antis:
        string = string.replace(anti, "")
    return string


def extract_parts(string: str, delims: Set[str]):
    if string[0] not in delims:
        delims.add("")

    pairs = [(string.index(d), d) for d in delims if d in string]
    pairs = sorted(pairs) + [(len(string), None)]

    parts = {}
    for i, (idx, delim) in enumerate(pairs[:-1]):
        next_idx = pairs[i + 1][0]
        parts[delim] = string[idx + len(delim) : next_idx]

    return parts


def inner_product(
    data: Tensor,
    patch_size: int = None,
    desc: str = "inner prod",
) -> Tensor:
    if not patch_size:
        return torch.inner(data, data)

    n_rows = data.shape[0]
    n_patches = (n_rows + patch_size - 1) // patch_size

    inner_prods = torch.zeros(n_rows, n_rows, device=data.device)
    for i in tqdm(range(n_patches), ncols=79, desc=desc):
        i0, i1 = i * patch_size, min((i + 1) * patch_size, n_rows)
        patch_i = data[i0:i1]
        for j in range(n_patches):
            j0, j1 = j * patch_size, min((j + 1) * patch_size, n_rows)
            patch_j = data[j0:j1]
            inner_prods[i0:i1, j0:j1] = torch.inner(patch_i, patch_j)
    return inner_prods
