from os.path import exists
from typing import Any, Optional, Tuple

from h5py import File
from neural_collapse.util import symm_reduce
from pandas import DataFrame, Index, read_csv
from torch import Tensor


def create_df(path: str) -> DataFrame:
    """Create a DataFrame or load it from <path> if it exists.

    Arguments:
        path (str): The filepath where the CSV dataframe will be saved. The
            ".csv" extension is appended to the path if not already there.

    Returns:
        DataFrame: A DataFrame with 'model' as the index, either loaded from
            existing data or empty.
    """

    if not path.endswith(".csv"):
        path = f"{path}.csv"
    if exists(path):
        df = read_csv(path, index_col="model")
    else:
        df = DataFrame(index=Index([], name="model"))
    for col in df.columns:
        if (df[col].fillna(0) % 1 == 0).all():
            df[col] = df[col].fillna(0).astype(int)
    return df


def update_df(df: DataFrame, metric: str, new_val: Any, entry: str):
    """Add or update a cell entry in the dataframe.

    Arguments:
        df (DataFrame): The DataFrame to update.
        metric (str): The label of the measurement (column name) to update.
        new_val (Any): The value to store, can be numeric or other types.
        entry (str): The index label for the row to update.

    Raises:
        AssertionError: If <new_val> is a Tensor and not a scalar.
    """
    if type(new_val) == Tensor:
        assert len(new_val.shape) == 0
        new_val = new_val.item()
    df.at[entry, metric] = new_val
    try:
        df[metric] = df[metric].astype(type(new_val))
    except:
        print(metric, df[metric].dtype, type(new_val))


def commit(path: str, metric: str, new_val: Any, entry: Optional[str] = None):
    """Update a PyTorch archive file at the specified <path>.

    Arguments:
        path (str): The location of the archive file (*.h5).
        metric (str): The label of the measurement (dataset name) to store.
        new_val (Any): The value to store, usually a Tensor.
        entry (Optional[str]): The index label for the dataset entry. Defaults
            to None, where the dataset will be replaced.

    Raises:
        Exception: If there are issues with file operations or tensor conversion.
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


def triu_stats(data: Tensor) -> Tuple[float, float, float]:
    """Compute basic statistics (mean, variance, standard deviation) of the
    upper-triangular values of a symmetric square matrix.

    Arguments:
        data (Tensor): A square matrix of input data (shape: (n, n)).

    Returns:
        Tuple[float, float, float]: A tuple containing the mean, variance,
            and standard deviation of the upper-triangular values.
    """
    mean = symm_reduce(data)
    var = symm_reduce(data, lambda row: ((row - mean) ** 2).sum())
    std = var**0.5

    return mean, var, std


def save_metrics(
    df: DataFrame,
    data: Tensor,
    key: str,
    iden: str,
    triu: bool = False,
    use_std: bool = False,
) -> Tuple[float, float]:
    """Compute basic statistics (mean, variance, standard deviation) of <data>
    and store the results in <df> for column <key> in row <iden>.

    Arguments:
        df (DataFrame): The DataFrame to update with computed statistics.
        data (Tensor): The input tensor for which statistics are computed.
        key (str): The label of the column where result will be stored.
        iden (str): The index label for the row where result will be stored.
        triu (bool, optional): Whether to compute statistics only for the
            upper-triangular values. Defaults to False.
        use_std (bool, optional): If True, standard deviation (instead of
            variance) will be computed and stored. Defaults to False.

    Returns:
        Tuple[float, float]: A tuple containing the mean and either standard deviation or variance
        (depending on the flags).
    """
    if triu:
        mean, var, std = triu_stats(data)
    else:
        mean, var, std = data.mean().item(), data.var().item(), data.std().item()

    update_df(df, f"{key}_mean", mean, iden)
    if use_std:
        update_df(df, f"{key}_std", std, iden)
    else:
        update_df(df, f"{key}_var", var, iden)

    return mean, (std if use_std else var)
