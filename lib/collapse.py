import os
from hashlib import sha256
from typing import List, Tuple, Union

import torch as pt
import torch.linalg as la
from torch import Tensor
from tqdm import tqdm

from lib.statistics import collect_hist, triu_mean, triu_std
from lib.utils import (frobenize, inner_product, log_kernel, normalize,
                       select_int_type)
from lib.visualization import plot_histogram

means_path = lambda name: f"means/{name}-means.pt"
vars_path = lambda name: f"vars/{name}-vars.pt"


class Statistics:
    def __init__(
        self,
        C: int = None,
        D: int = None,
        device: Union[str, pt.device] = "cpu",
        dtype: pt.dtype = pt.float32,
        load_means: str = None,
        load_vars: str = None,
        verbose: bool = True,
    ):
        """
        C: number of classes
        D: dimension of embeddings
        device: which processor to put the tensors on
        dtype: floating point precision
        load_means: file from which to load a means checkpoint
        load_vars: file from which to load a covariances checkpoint
        """

        self.C, self.D = C, D
        self.device = device
        self.dtype, self.eps = dtype, pt.finfo(dtype).eps
        self.hash = None  # ensuring covariances based on consistent means

        if load_means:
            self.load_totals(load_means, verbose)
        if load_vars:
            self.load_var_sums(load_vars, verbose)

        self.ctype = select_int_type(self.C)
        if load_means and load_vars:
            return

        if not load_means:
            self.N1, self.N1_seqs = 0, 0  # counts of samples/sequences for means pass
            self.counts = pt.zeros(self.C, dtype=pt.int32).to(device)
            self.totals = pt.zeros(self.C, self.D, dtype=dtype).to(device)

        if not load_vars:
            self.N2, self.N2_seqs = 0, 0  # counts of samples/sequences for vars pass
            self.var_sums = pt.zeros(self.C, dtype=dtype).to(device)

    def move_to(self, device: str = "cpu"):
        self.counts = self.counts.to(device)
        self.totals = self.counts.to(device)
        self.var_sums = self.var_sums.to(device)

    def counts_in_range(self, minimum: int = 0, maximum: int = None):
        idxs = self.counts.squeeze() >= minimum
        assert pt.all(minimum <= self.counts[idxs])
        if maximum:
            idxs &= self.counts.squeeze() < maximum
            assert pt.all(self.counts[idxs] < maximum)

        filtered = idxs.nonzero().squeeze()

        return filtered

    def collect_means(self, X: Tensor, Y: Tensor, B: int = 1) -> Tuple[Tensor, Tensor]:
        """First pass: increment vector counts and totals for a batch.
        B: batch size
        X (B x D): feature vectors
        Y (B x 1): class labels
        """

        if len(Y.shape) < 1:
            print("W: batch too short")
            return None, None
        if len(Y.shape) > 1:
            Y = pt.squeeze(Y)
        Y = Y.to(self.ctype)

        assert X.shape[0] == Y.shape[0]
        self.N1 += Y.shape[0]
        self.N1_seqs += B

        label_range = pt.arange(self.C, dtype=self.ctype)
        idxs = Y[:, None] == label_range.to(Y.device)
        X, idxs = X.to(self.device), idxs.to(self.device)

        self.counts += pt.sum(idxs, dim=0, dtype=self.ctype)[:, None].squeeze()  # C
        self.totals += pt.matmul(idxs.mT.to(self.dtype), X.to(self.dtype))  # C x D

        return self.counts, self.totals

    def collect_vars(self, X: Tensor, Y: Tensor, B: int = 1) -> Tuple[Tensor, Tensor]:
        """Second pass: increment within/between-class covariance for a batch.
        B: batch size
        X (B x D): feature vectors
        Y (B x 1): class labels
        """

        idxs = self.counts_in_range()
        means, mean_G = self.compute_means(idxs)

        assert X.shape[-1] == self.D
        assert X.shape[0] == Y.shape[0]

        if self.N2 + Y.shape[0] > self.N1:
            print("  W: this batch would exceed means samples")
            print(f"  {self.N2}+{Y.shape[0]} > {self.N1}")

        self.N2 += Y.shape[0]
        self.N2_seqs += B

        diffs = X.to(self.device) - means[Y]
        Y = Y.to(self.device).to(pt.int64)
        self.var_sums.scatter_add_(0, Y, pt.sum(diffs * diffs, dim=-1))

        return self.var_sums

    def compute_means(
        self, idxs: List[int] = None, weighted: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Compute and store the overall means of the dataset.
        idxs: classes to select for subsampled computation.
        weighted: whether to use the global mean (True) as opposed to the
            unweighted average mean (False).
        """

        counts = (self.counts if idxs is None else self.counts[idxs]).unsqueeze(-1)
        N = counts.sum()
        if idxs is None:
            assert N == self.N1

        totals = self.totals if idxs is None else self.totals[idxs]
        mean = totals / (counts + self.eps).to(self.dtype)  # C' x D
        if weighted:
            mean_G = counts.T.to(self.dtype) @ mean / (N + self.eps)  # D
        else:
            mean_G = mean.mean(dim=0)

        self.hash = sha256(mean.cpu().numpy().tobytes()).hexdigest()

        return mean, mean_G

    def compute_vars(self, idxs: List[int] = None) -> Tensor:
        """Compute the overall covariances of the dataset.
        idxs: classes to select for subsampled computation.
        """

        if self.N1 != self.N2 or self.N1_seqs != self.N2_seqs:
            return None

        counts, var_sums = self.counts, self.var_sums
        if idxs is not None:
            counts, var_sums = counts[idxs], var_sums[idxs]
        C = len(counts)
        vars_normed = var_sums / counts

        means, _ = self.compute_means(idxs)
        CDNVs = pt.zeros(C, C, dtype=self.dtype, device=self.device)
        for c in tqdm(range(C), ncols=79, desc="cdnvs"):
            var_avgs = (vars_normed[c] + vars_normed).squeeze() / 2
            means_diff = means[c] - means
            inner = pt.sum(means_diff * means_diff, dim=-1)
            CDNVs[c] = var_avgs.squeeze(0) / inner

        return CDNVs

    ## COLLAPSE MEASURES ##

    def mean_norms(self, idxs: List[int] = None) -> Tensor:
        """Compute norms of class means for measure equinormness of NC2/GCN2.
        idxs: classes to select for subsampled computation.
        """
        means, _ = self.compute_means(idxs)
        norms = means.norm(dim=-1) ** 2  # C'

        return norms

    def interference(self, idxs: List[int] = None, patch_size: int = None) -> Tensor:
        """Compute interference between class means to measure convergence to
        a simplex ETF (NC2), which is equivalent to the identity matrix.
        idxs: classes to select for subsampled computation.
        patch_size: the max patch size for memory-constrained environments.
        """

        means, mean_G = self.compute_means(idxs)
        mean_diffs = means - mean_G  # C' x D

        interference = inner_product(normalize(mean_diffs), patch_size, "interfere")

        return interference  # C' x C'

    def geodesic_surplus(
        self, idxs: List[int] = None, kernel: callable = log_kernel
    ) -> Tensor:
        """Compute geodesic surpluses towards to measure convergence to
        hyperspherical uniformity (GNC2, https://arxiv.org/abs/2303.06484).
        idxs: classes to select for subsampled computation.
        kernel: how to compute geodesic distance between points.
        """
        means, _ = self.compute_means(idxs)
        dists = kernel(means)  # C' x C'
        surplus = dists - dists.min()

        return surplus  # C' x C'

    def diff_duality(self, weights: Tensor, idxs: List[int] = None) -> pt.float:
        """Compute self-duality (NC3).
        weights (C x D): weights of the linear classifier
        idxs: classes to select for subsampled computation.
        """
        means, mean_G = self.compute_means(idxs)
        means_frobed = frobenize(means - mean_G).to(self.device) # C x D
    
        weights = weights if idxs is None else weights[idxs]
        weights_frobed = frobenize(weights).to(self.device) # C x D

        duality = la.norm(weights_frobed - means_frobed) # 1
        
        return duality.cpu()

    def dot_duality(
        self,
        weights: Tensor,
        idxs: List[int] = None,
    ) -> Tensor:
        """Compute dot-product similarities between means and classifier.
        The average of this matrix is linearly related to self-duality (NC3).
        weights (C x D): weights of the linear classifier
        idxs: classes to select for subsampled computation.
        dims: dimensions on which to project the weights and mean norms
        """
        means, mean_G = self.compute_means(idxs)
        means_frobed = frobenize(means - mean_G).to(self.device) # C x D
    
        weights = weights if idxs is None else weights[idxs]
        weights_frobed = frobenize(weights).to(self.device) # C x D

        dot_prod = means_frobed * weights_frobed  # C x D
        result = pt.sum(dot_prod, dim=1)  # C

        return result

    def project_classes(
        self,
        weights: Tensor,
        idxs: List[int] = None,
        dims: Tuple[int] = None,
    ):
        means, mean_G = self.compute_means(idxs)
        means_frobed = frobenize(means - mean_G).to(self.device) # C x D
    
        weights = weights if idxs is None else weights[idxs]
        weights_frobed = frobenize(weights).to(self.device) # C x D

        selected = (weights_frobed[dims, :] + means_frobed[dims, :]) / 2
        proj_means = selected @ means_frobed.mT
        proj_cls = selected @ weights_frobed.mT
        return proj_means, proj_cls

    ## SAVING AND LOADING ##

    def save_totals(self, file: str, verbose: bool = False) -> str:
        self.compute_means()
        data = {
            "hash": self.hash,
            "C": self.C,
            "D": self.D,
            "N": self.N1,
            "N_seqs": self.N1_seqs,
            "counts": self.counts,
            "totals": self.totals,
        }
        pt.save(data, file)

        if verbose:
            print(f"SAVED means to {file}; {self.N1} in {self.N1_seqs} seqs")
            print(f"  HASH: {self.hash}")
        return file

    def load_totals(self, file: str, verbose: bool = True) -> int:
        if not os.path.isfile(file):
            print(f"  W: file {file} not found; need to collect means from scratch")
            return 0

        data = pt.load(file, self.device)
        assert self.hash in [None, data["hash"]], "overwriting current data"
        self.hash = data["hash"]

        self.C, self.D = data["C"], data["D"]
        self.N1 = data["N"]
        self.N1_seqs = data["N_seqs"]
        self.counts = data["counts"].to(self.device).squeeze()
        self.totals = data["totals"].to(self.device)

        if verbose:
            print(f"LOADED means from {file}; {self.N1} in {self.N1_seqs} seqs")
        return self.N1_seqs

    def save_var_sums(self, file: str, verbose: bool = False) -> str:
        data = {
            "hash": self.hash,
            "C": self.C,
            "D": self.D,
            "N": self.N2,
            "N_seqs": self.N2_seqs,
            "var_sums": self.var_sums,
        }
        pt.save(data, file)

        if verbose:
            print(f"SAVED vars to {file}; {self.N2} in {self.N2_seqs} seqs")
            print(f"  MEANS HASH: {self.hash}")
        return file

    def load_var_sums(self, file: str, verbose: bool = True) -> int:
        means_file = file.replace("vars", "means")
        if not self.load_totals(means_file, False):
            print("  W: means not found; please collect them first")
            return 0

        if not os.path.isfile(file):
            print(f"  W: path {file} not found; need to collect vars from scratch")
            return 0

        data = pt.load(file, self.device)
        assert data["hash"] == self.hash, "vars based on outdated means"

        self.C, self.D = data["C"], data["D"]
        self.N2 = data["N"]
        self.N2_seqs = data["N_seqs"]
        self.var_sums = data["var_sums"].to(self.device)

        if verbose:
            print(f"LOADED vars from {file}; {self.N2} in {self.N2_seqs} seqs")
        return self.N2_seqs


if __name__ == "__main__":
    pt.set_grad_enabled(False)

    import os
    from math import ceil, log
    from sys import argv

    import matplotlib.pyplot as plt

    means_file, vars_file = "dummy-means.pt", "dummy-vars.pt"
    if os.path.exists(means_file):
        os.remove(means_file)
    if os.path.exists(vars_file):
        os.remove(vars_file)

    matshow_path, hist_path = "cdnvs_mat.png", "cdnvs_hist.png"
    if os.path.exists(matshow_path):
        os.remove(matshow_path)
    if os.path.exists(hist_path):
        os.remove(hist_path)

    dtype = pt.float32
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    C, D, N, B = (2, 3, 12, 4) if len(argv) < 5 else [int(arg) for arg in argv[1:5]]
    if N % B != 0:
        N += B - N % B
        print(f"info: batch_size is {B}, so rounding up to {N}")
    if N < ceil(-C * log(0.01)):
        print(f"W: {N} samples may not be enough...")

    Y = pt.randint(C, (N,), dtype=pt.int32)
    X = (pt.randn(N, D, dtype=dtype) + pt.randn(C, D)[Y, :]).to(device)
    print(f"created: X {X.shape}, Y {Y.shape}")

    stats = Statistics(C, D, device=device, dtype=dtype)
    print("collecting means")
    for i in range(0, N, B):
        batch, labels = X[i : i + B], Y[i : i + B]
        stats.collect_means(batch, labels, B)

    stats.save_totals(means_file)
    stats.load_totals(means_file)
    W = pt.randn(C, D, dtype=dtype)
    duality = stats.diff_duality(W)
    print("self-duality:", duality)

    coh = stats.coherence()
    hist, edges = collect_hist(coh, triu=True, desc="coh hist")
    print("coherence:", triu_std(coh).cpu() / triu_mean(coh).cpu())

    ## Variances ##

    print("collecting vars")
    for i in range(0, N - B, B):
        batch, labels = X[i : i + B], Y[i : i + B]
        stats.collect_vars(batch, labels, B)
    stats.save_var_sums(vars_file)
    stats.load_var_sums(vars_file)
    stats.collect_vars(X[-B:], Y[-B:], B)

    # class-distance normalized variances (CDNVS)
    CDNVs = stats.compute_vars()

    plt.matshow(CDNVs.cpu())
    plt.colorbar()
    plt.savefig(matshow_path)
    plt.close()

    hist, edges = collect_hist(CDNVs, 1024, True)
    fig, ax = plt.subplots(figsize=(3, 2))
    plot_histogram(ax, hist, edges, "dummy", "#000000")
    fig.savefig(hist_path)

    mean = triu_mean(CDNVs)
    std = triu_std(CDNVs, mean)
    inv_snr = std.cpu() / mean.cpu()
    print("inverse SNR:", inv_snr)
