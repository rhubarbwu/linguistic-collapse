import gc
from math import copysign
from os.path import isfile
from typing import Any, List, Set, Tuple

import torch as pt
import torch.linalg as la
from torch import (Tensor, bfloat16, float16, float32, int8, int16, int32,
                   int64, uint8)
from tqdm import tqdm

frobenize = lambda x: x / la.norm(x)
normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + pt.finfo(x.dtype).eps)
get_dev_name = lambda d: "cpu" if d == "cpu" else pt.cuda.get_device_name(d)


DTYPES = {
    "fp32": float32,
    "fp16": float16,
    "bf16": bfloat16,
}


def select_int_type(value: int = 0) -> pt.dtype:
    if value is None:
        return None
    for dtype in [int8, uint8, int16, int32, int64]:
        if value < pt.iinfo(dtype).max:
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
    if pt.cuda.is_available():
        pt.cuda.empty_cache()


CRUFT = ["TS", ".pt", "means-", "vars-", "-means", "-vars"]


def identify(path: str) -> str:
    identifier = path.split("/")[-1]
    cruft_matches = 0
    for cruft in CRUFT:
        if cruft in identifier:
            cruft_matches += 1
        identifier = identifier.replace(cruft, "")
    return identifier


def pathify(identifier: str, stats_dir: str) -> Tuple[str, str]:
    means_path = f"{stats_dir}/means-{identifier}.pt"
    if not isfile(means_path):
        means_path = f"{stats_dir}/{identifier}-means.pt"
        if not isfile(means_path):
            return None, None
    means_path = means_path.replace("//", "/")

    vars_path = f"{stats_dir}/vars-{identifier}.pt"
    if not isfile(vars_path):
        vars_path = f"{stats_dir}/{identifier}-vars.pt"
        if not isfile(vars_path):
            vars_path = None

    vars_path = None if vars_path is None else vars_path.replace("//", "/")

    return means_path, vars_path


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


def log_kernel(matrix: Tensor, device: str = "cpu"):
    N = len(matrix)
    normed = normalize(matrix)
    kernel_grid = pt.zeros((N, N), device=device)
    for idx in tqdm(range(N), ncols=79, desc="log kernel"):
        diff_norms = (normed[idx] - normed).norm(dim=-1) # normalize first
        kernel_grid[idx] = (diff_norms ** (-1)).log()

    return kernel_grid


def riesz_kernel(matrix: Tensor, device: str = "cpu"):
    N, S = len(matrix), matrix.shape[-1] - 2
    normed = normalize(matrix)
    kernel_grid = pt.zeros((N, N), device=device)
    for idx in tqdm(range(N), ncols=79, desc="riesz kernel"):
        diff_norms = (normed[idx] - normed).norm(dim=-1)
        kernel_grid[idx] = copysign(1, S) * diff_norms ** (-S)

    return kernel_grid


def inner_product(
    data: Tensor, patch_size: int = None, desc: str = "inner prod"
) -> Tensor:
    if not patch_size:
        return pt.inner(data, data)

    n_rows = data.shape[0]
    n_patches = (n_rows + patch_size - 1) // patch_size

    inner_prods = pt.zeros(n_rows, n_rows, device=data.device)
    for i in tqdm(range(n_patches), ncols=79, desc=desc):
        i0, i1 = i * patch_size, min((i + 1) * patch_size, n_rows)
        patch_i = data[i0:i1]
        for j in range(n_patches):
            j0, j1 = j * patch_size, min((j + 1) * patch_size, n_rows)
            patch_j = data[j0:j1]
            inner_prods[i0:i1, j0:j1] = pt.inner(patch_i, patch_j)
    return inner_prods
