import gc
from math import isnan
from os.path import isfile
from typing import Any, List, Set, Tuple

import torch as pt
import torch.linalg as la
from torch import bfloat16, float16, float32, int8, int16, int32, int64, uint8

frobenize = lambda x: x / la.norm(x)
normalize = lambda x: x / (la.norm(x, dim=-1, keepdim=True) + pt.finfo(x.dtype).eps)
get_dev_name = lambda d: "cpu" if d == "cpu" else pt.cuda.get_device_name(d)


DTYPES = {
    "fp32": float32,
    "fp16": float16,
    "bf16": bfloat16,
}


def select_int_type(value: int = 0) -> pt.dtype:
    """Return the smallest integer type to store <value>."""
    if value is None:
        return None
    for dtype in [int8, uint8, int16, int32, int64]:
        if value < pt.iinfo(dtype).max:
            return dtype

    raise ValueError(f"{value} too big")


def is_float(value: Any):
    """Check if <value> is a float."""
    try:
        float(value)
        return not isnan(value)
    except ValueError:
        return False


MAGS = ["q", "t", "b", "m", "k"]
MAG_STRS = ["quadrillion", "trillion", "billion", "million", "thousand"]


def numerate(size: str, ref: str = "k") -> float:
    """Convert size notation to numerical integer.
    size: original size notation (e.g. 5m).
    ref: benchmark/reference scale.
    """
    ref_idx = MAGS.index(ref)
    size = size.split("x")[0].lower()

    for i, scale in enumerate(MAGS[:-1]):
        if scale in size:
            num = float(size.replace(scale, ""))
            scaled = num * 1000 ** (ref_idx - i)
            return scaled


def clean_up(*garbage: List[Any]):
    """Simple CPU/CUDA <garbage> collection."""
    for g in garbage:
        del g
    gc.collect()
    if pt.cuda.is_available():
        pt.cuda.empty_cache()


CRUFT = ["TS", "TinyStories-", ".pt", "means-", "vars-", "-means", "-vars", "-decs"]


def identify(path: str) -> str:
    """Extract model/experiment indentifier from <path>."""
    identifier = path.split("/")[-1]
    cruft_matches = 0
    for cruft in CRUFT:
        if cruft in identifier:
            cruft_matches += 1
        identifier = identifier.replace(cruft, "")
    return identifier


def pathify(identifier: str, stats_dir: str) -> Tuple[str, str]:
    """Construct means and vars file paths within <stats_dir> from <identifier>"""
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
    """Remove extraneous parts <antis> from <string>."""
    for anti in antis:
        string = string.replace(anti, "")
    return string


def extract_parts(string: str, delims: Set[str]):
    """Split <string> into parts separated by <delims>."""
    if string[0] not in delims:
        delims.add("")

    pairs = [(string.index(d), d) for d in delims if d in string]
    pairs = sorted(pairs) + [(len(string), None)]

    parts = {}
    for i, (idx, delim) in enumerate(pairs[:-1]):
        next_idx = pairs[i + 1][0]
        parts[delim] = string[idx + len(delim) : next_idx]

    return parts
