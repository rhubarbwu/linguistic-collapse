from typing import Tuple

import torch as pt
import torch.linalg as la
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torch import Tensor
from tqdm import tqdm


def apply_pca(data: Tensor, K: int = 2) -> Tensor:
    cov_matrix = pt.mm(data.T, data) / (data.shape[0] - 1)

    eig_vals, eig_vecs = la.eig(cov_matrix)
    eig_vals, eig_vecs = pt.real(eig_vals), pt.real(eig_vecs)

    eig_val_idx = pt.argsort(eig_vals, descending=True)
    eig_vals = eig_vals[eig_val_idx]
    eig_vecs = eig_vecs[:, eig_val_idx]

    projected = pt.mm(data, eig_vecs[:, :K])
    return projected


def cluster_gmm(data, K):
    gmm = GaussianMixture(n_components=K, random_state=0)
    labels = gmm.fit_predict(data.cpu())
    means = gmm.means_
    del gmm
    return labels, means


def cluster_kmeans(data, K):
    KM = KMeans if data.shape[0] < 16384 else MiniBatchKMeans

    kmeans = KM(n_clusters=K, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(data.cpu())
    centres = kmeans.cluster_centers_
    del kmeans
    return labels, centres


def cluster_spectral(data, K):
    spectral = SpectralClustering(n_clusters=K)
    labels = spectral.fit_predict(data)
    del spectral
    return labels


def compute_centroids(
    data: Tensor,
    K: int = 1,
    weights: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    assert data.shape[0] == weights.shape[0]

    if K == 1:
        labels = pt.zeros(len(weights), dtype=pt.uint8, device=data.device)
    elif K > 1:
        labels, centroids = cluster_kmeans(data, K)
        if weights is None:  # unweighted centroids
            return labels, pt.tensor(centroids, device=data.device)

    eps = pt.finfo(data.dtype).eps

    # weighted centroids
    centroids = pt.zeros(K, data.shape[-1], device=data.device)
    for r in range(K):
        data_r = data[labels == r]
        weights_r = weights[labels == r].to(data.dtype)
        total_r = weights_r.sum() + eps
        centroids[r] = weights_r.mT @ data_r / total_r

    return labels, centroids


def meanify_diag(data: Tensor) -> Tensor:
    assert pt.all(pt.tensor(data.shape) == data.shape[0])
    data.fill_diagonal_(0)
    mean = data.sum() / (data.shape[0] * (data.shape[0] - 1))
    data.fill_diagonal_(mean)
    return data


def collect_hist(
    data: Tensor,
    num_bins: int = 64,
    triu: bool = False,
    desc: str = "histogram",
) -> Tuple[Tensor, Tensor]:
    N = data.shape[0]
    min_val, max_val = data.min(), data.max()
    val_range = max_val - min_val
    min_val -= 0.01 * val_range
    max_val += 0.01 * val_range

    hist = pt.zeros(num_bins, dtype=int, device=data.device)
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
