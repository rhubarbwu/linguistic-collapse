from os.path import exists
from typing import Any, Optional, Tuple

import pandas as pd
import torch as pt
import torch.linalg as la
from h5py import File
from torch import Tensor
from tqdm import tqdm

from lib.utils import select_int_type


def apply_pca(data: Tensor, K: int = 2) -> Tensor:
    cov_matrix = pt.mm(data.T, data) / (data.shape[0] - 1)

    eig_vals, eig_vecs = la.eig(cov_matrix)
    eig_vals, eig_vecs = pt.real(eig_vals), pt.real(eig_vecs)

    eig_val_idx = pt.argsort(eig_vals, descending=True)
    eig_vals = eig_vals[eig_val_idx]
    eig_vecs = eig_vecs[:, eig_val_idx]

    projected = pt.mm(data, eig_vecs[:, :K])
    return projected


def triu_mean(data: Tensor, desc: str = "means") -> pt.float:
    N = data.shape[0]
    total = 0

    assert N == data.shape[1]
    for i in tqdm(range((N - 1) // 2), ncols=79, desc=desc):
        upper = data[i][i + 1 :]
        lower = data[N - i - 2][N - i - 1 :]
        folded = pt.cat((upper, lower))
        total += folded.sum()
    if N % 2 == 0:
        row = data[N // 2 - 1][N // 2 :]
        total += row.sum()

    mean = total / (N * (N - 1) / 2)
    return mean


def triu_std(
    data: Tensor, mean: pt.float = None, correction: bool = True, desc: str = "std"
) -> pt.float:
    debias = 1 if correction else 0
    if mean is None:
        mean = triu_mean(data)

    N = data.shape[0]
    total = 0

    assert N == data.shape[1]
    for i in tqdm(range((N - 1) // 2), ncols=79, desc=desc):
        upper = data[i][i + 1 :]
        lower = data[N - i - 2][N - i - 1 :]
        folded = pt.cat((upper, lower))
        total += ((folded - mean) ** 2).sum()
    if N % 2 == 0:
        row = data[N // 2 - 1][N // 2 :]
        total += ((row - mean) ** 2).sum()

    var = total / (N * (N - 1) / 2 - debias)
    return var.sqrt()


def collect_hist(
    data: Tensor, num_bins: int = 64, triu: bool = False, desc: str = "histogram"
) -> Tuple[Tensor, Tensor]:
    N = data.shape[0]
    min_val, max_val = data.min(), data.max()
    val_range = max_val - min_val
    min_val -= 0.01 * val_range
    max_val += 0.01 * val_range

    hist = pt.zeros(num_bins, dtype=select_int_type(data.numel()), device=data.device)
    count = lambda x: pt.histc(x, num_bins, min_val, max_val).int()
    if triu:
        for i in tqdm(range((N - 1) // 2), ncols=79, desc=desc):
            upper = data[i][i + 1 :]
            lower = data[N - i - 2][N - i - 1 :]
            folded = pt.cat((upper, lower))
            hist += count(folded)
        if N % 2 == 0:
            row = data[N // 2 - 1][N // 2 :]
            hist += count(row)
    else:
        for row in tqdm(data, ncols=79, desc=desc):
            hist += count(row)

    edges = pt.linspace(min_val, max_val, num_bins + 1)

    return hist, edges


def create_df(path: str) -> pd.DataFrame:
    path = f"{path}.csv"
    if exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame({"model": []})
    df = df.set_index("model")
    return df


def update_df(df: pd.DataFrame, metric: str, new_val: Any, entry: Optional[str] = None):
    if entry:
        if type(new_val) == Tensor:
            assert len(new_val.shape) == 0
            new_val = new_val.item()
        if metric not in df:
            df[metric] = pd.Series(dtype=type(new_val))
        df.at[entry, metric] = new_val


def commit(path: str, metric: str, new_val: Any, entry: Optional[str] = None):
    with File(f"{path}.h5", "a") as file:
        if new_val is not None and entry is not None:
            if metric not in file:
                file.create_group(metric)
            if entry in file[metric]:
                del file[metric][entry]
            try:
                file[metric][entry] = new_val.cpu()
            except AttributeError:
                file[metric][entry] = new_val
        elif new_val is not None:
            if metric in file:
                del file[metric]
            file.create_dataset(metric, data=new_val.cpu())
