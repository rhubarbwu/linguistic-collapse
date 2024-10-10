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
    """Conduct PCA on <data> to <K> dimensions."""
    cov_matrix = pt.mm(data.T, data) / (data.shape[0] - 1)

    eig_vals, eig_vecs = la.eig(cov_matrix)
    eig_vals, eig_vecs = pt.real(eig_vals), pt.real(eig_vecs)

    eig_val_idx = pt.argsort(eig_vals, descending=True)
    eig_vals = eig_vals[eig_val_idx]
    eig_vecs = eig_vecs[:, eig_val_idx]

    projected = pt.mm(data, eig_vecs[:, :K])
    return projected


def triu_mean(data: Tensor, desc: str = "means") -> float:
    """Compute the mean of the upper triangle in <data>."""
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
    data: Tensor, mean: float = None, correction: bool = True, desc: str = "std"
) -> float:
    """Compute the std of the upper triangle in <data>."""
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


def create_df(path: str) -> pd.DataFrame:
    """Create CSV dataframe at <path> if it doesn't exist."""
    path = f"{path}.csv"
    if exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame({"model": []})
    df = df.set_index("model")
    return df


def update_df(df: pd.DataFrame, metric: str, new_val: Any, entry: str):
    """Add a cell entry to the CSV dataframe.
    df: Dataframe.
    metric: label of measurement.
    new_val: numerical (or otherwise) value to store.
    entry: index label.
    """
    if type(new_val) == Tensor:
        assert len(new_val.shape) == 0
        new_val = new_val.item()
    if metric not in df:
        df[metric] = pd.Series(dtype=type(new_val))
    df.at[entry, metric] = new_val


def commit(path: str, metric: str, new_val: Any, entry: Optional[str] = None):
    """Update a PyTorch archive file.
    path: location of archive file (*.h5).
    metric: label of measurement.
    new_val: numerical (or otherwise) values (usually a Tensor) to store.
    entry: index label.
    """
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
