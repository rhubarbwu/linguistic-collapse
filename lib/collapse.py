import os
from hashlib import sha256
from typing import List, Tuple, Union

import torch as pt
import torch.linalg as la
from torch import Tensor

from lib.utils import inner_product, normalize, select_int_type

means_dir = lambda prefix: f"{prefix}-means.pt"
covs_dir = lambda prefix: f"{prefix}-covs.pt"


class Statistics:
    def __init__(
        self,
        C: int = None,
        D: int = None,
        device: Union[str, pt.device] = "cpu",
        dtype: pt.dtype = pt.float32,
        load_means: str = None,
        load_covs: str = None,
        verbose: bool = True,
    ):
        """
        C: number of classes
        D: dimension of embeddings
        device: which processor to put the tensors on
        dtype: floating point precision
        load_means: file from which to load a means checkpoint
        load_covs: file from which to load a covariances checkpoint
        """

        self.C, self.D = C, D
        self.device = device
        self.dtype, self.eps = dtype, pt.finfo(dtype).eps
        self.hash = None  # ensuring covariances based on consistent means

        if load_means:
            self.load_totals(load_means, verbose)
        if load_covs:
            self.load_covs_sums(load_covs, verbose)

        self.ctype = select_int_type(self.C)
        if load_means and load_covs:
            return

        if not load_means:
            self.N1, self.N1_seqs = 0, 0  # counts of samples/sequences for means pass
            self.counts = pt.zeros(self.C, dtype=self.ctype).to(device)
            self.totals = pt.zeros(self.C, self.D, dtype=dtype).to(device)

        if not load_covs:
            self.N2, self.N2_seqs = 0, 0  # counts of samples/sequences for covs pass
            self.sum_between = pt.zeros(self.D, self.D, dtype=dtype).to(device)
            self.sum_within = pt.zeros(self.D, self.D, dtype=dtype).to(device)

    def move_to(self, device: str = "cpu"):
        self.counts = self.counts.to(device)
        self.totals = self.counts.to(device)
        self.sum_between = self.sum_between.to(device)
        self.sum_within = self.sum_within.to(device)

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

    def collect_covs(self, X: Tensor, Y: Tensor, B: int = 1) -> Tuple[Tensor, Tensor]:
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

        mean_diff = means[Y] - mean_G  # C' x D
        self.sum_between += pt.matmul(mean_diff.mT, mean_diff)  # D x D

        diff = X.to(self.device) - means[Y]  # N x D
        self.sum_within += pt.matmul(diff.mT, diff)  # D x D

        return self.sum_between, self.sum_within

    def report_stats(self, idxs: List[int] = None, covs: bool = False):
        means, mean_G = self.compute_means(idxs)
        print("means count", self.N1)
        print("class means", means)
        print("global mean", mean_G)

        if covs:  # second pass
            print("              covs count", self.N2)
            cov_between, cov_within = self.compute_covs()
            print("between-class covariance", cov_between)
            print(" within-class covariance", cov_within)

    def compute_means(self, idxs: List[int] = None) -> Tuple[Tensor, Tensor]:
        """Compute and store the overall means of the dataset."""

        counts = (self.counts if idxs is None else self.counts[idxs]).unsqueeze(-1)
        N = counts.sum()
        if idxs is None:
            assert N == self.N1

        totals = self.totals if idxs is None else self.totals[idxs]
        self.means = totals / (counts + self.eps).to(self.dtype)  # C' x D
        self.mean_G = counts.T.to(self.dtype) @ self.means / (N + self.eps)  # D
        self.hash = sha256(self.means.cpu().numpy().tobytes()).hexdigest()

        return self.means, self.mean_G

    def compute_covs(self) -> Tuple[Tensor, Tensor]:
        """Compute the overall covariances of the dataset."""

        if self.N1 != self.N2 or self.N1_seqs != self.N2_seqs:
            return None, None

        cov_between = self.sum_between / (self.N2 + self.eps)  # D x D
        cov_within = self.sum_within / (self.N2 + self.eps)  # D x D

        return cov_between, cov_within

    ## COLLAPSE MEASURES ##

    def coherence(
        self, idxs: List[int] = None, K: int = 1, patch_size: int = None
    ) -> Tensor:
        """Compute coherence between class means.
        idxs: classes to select for subsampled computation.
        K: number of cluster (regional) means to centre around.
        patch_size: if given, the maximum patch size for memory-constrained environments.
        """

        means, mean_G = self.compute_means(idxs)
        diff = means - mean_G

        return inner_product(normalize(diff), patch_size, "coh inner")

    def diff_duality(self, weights: Tensor, idxs: List[int] = None) -> pt.float:
        """Compute self-duality (NC3).
        weights (C x D): weights of the linear classifier
        K: number of cluster (regional) means to centre around.
        idxs: classes to select for subsampled computation.
        """
        means, mean_G = self.compute_means(idxs)
        diff = means - mean_G

        weights = weights if idxs is None else weights[idxs]
        weights_normed = normalize(weights).to(self.device)

        duality = la.norm(normalize(diff) - weights_normed) ** 2 / self.C
        return duality.cpu()

    def dot_duality(
        self,
        weights: Tensor,
        idxs: List[int] = None,
        dims: Tuple[int] = (0, 1),
    ) -> Tensor:
        means, mean_G = self.compute_means(idxs)
        diff_normed = normalize(means - mean_G)

        weights = weights if idxs is None else weights[idxs]
        weights_normed = normalize(weights).to(self.device)

        dot_prod = diff_normed * weights_normed
        result = pt.sum(dot_prod, dim=1)

        selected = (weights_normed[dims, :] + diff_normed[dims, :]) / 2
        proj_means = selected @ diff_normed.mT
        proj_cls = selected @ weights_normed.mT

        return result, proj_means, proj_cls

    def inv_snr(self) -> pt.float:
        """Compute separation fuzziness/signal-to-noise ratio tr{Sw Sb^-1} (NC2)."""
        if self.N2 == 0:
            print("  W: no covariances!")
            return None

        cov_between, cov_within = self.compute_covs()  # D x D, D x D
        if cov_between is None or cov_within is None:
            print("  W: means and covariances counts don't match!")
            return None

        cov_between_inv = la.pinv(cov_between)
        snr_inv = pt.trace(cov_within @ cov_between_inv)  # 0 (scalar)
        return snr_inv.cpu()

    ## SAVING AND LOADING ##

    def save_totals(self, file: str = "untitled", verbose: bool = False) -> str:
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
        if not os.path.isfile(file):
            file = means_dir(file)
        pt.save(data, file)

        if verbose:
            print(f"SAVED means to {file}; {self.N1} in {self.N1_seqs} seqs")
            print(f"  HASH: {self.hash}")
        return file

    def load_totals(self, file: str = "untitled", verbose: bool = True) -> int:
        if not os.path.isfile(file):
            file = means_dir(file)
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

    def save_covs_sums(self, file: str = "untitled", verbose: bool = False) -> str:
        data = {
            "hash": self.hash,
            "C": self.C,
            "D": self.D,
            "N": self.N2,
            "N_seqs": self.N2_seqs,
            "sum_between": self.sum_between,
            "sum_within": self.sum_within,
        }
        if not os.path.isfile(file):
            file = covs_dir(file)
        pt.save(data, file)

        if verbose:
            print(f"SAVED covs to {file}; {self.N2} in {self.N2_seqs} seqs")
            print(f"  MEANS HASH: {self.hash}")
        return file

    def load_covs_sums(self, file: str = "untitled", verbose: bool = True) -> int:
        if not os.path.isfile(file):
            file = covs_dir(file)

        means_file = file.replace("covs", "means")
        if not self.load_totals(means_file, False):
            print("  W: means not found; please collect them first")
            return 0

        if not os.path.isfile(file):
            print(f"  W: path {file} not found; need to collect covs from scratch")
            return 0

        data = pt.load(file, self.device)
        assert data["hash"] == self.hash, "covs based on outdated means"

        self.C, self.D = data["C"], data["D"]
        self.N2 = data["N"]
        self.N2_seqs = data["N_seqs"]
        self.sum_between = data["sum_between"].to(self.device)
        self.sum_within = data["sum_within"].to(self.device)

        if verbose:
            print(f"LOADED covs from {file}; {self.N2} in {self.N2_seqs} seqs")
        return self.N2_seqs


if __name__ == "__main__":
    from math import ceil, log
    from sys import argv

    dtype = pt.float32
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    C, D, N, B = (2, 3, 12, 4) if len(argv) < 5 else [int(arg) for arg in argv[1:5]]
    if N % B != 0:
        N += B - N % B
        print(f"info: batch_size is {B}, so rounding up to {N}")
    if N < ceil(-C * log(0.01)):
        print(f"W: {N} samples may not be enough...")

    Y = pt.randint(C, (N,))
    X = (pt.randn(N, D, dtype=dtype) + pt.randn(C, D)[Y, :]).to(device)
    print(f"created: X {X.shape}, Y {Y.shape}")

    stats = Statistics(C, D, device=device, dtype=dtype)
    print("collecting means")
    for i in range(0, N, B):
        batch, labels = X[i : i + B], Y[i : i + B]
        stats.collect_means(batch, labels, B)

    stats.save_totals()
    stats.load_totals()
    stats.compute_means()  # would be implicitly called in compute_covs()

    print("collecting covs")
    for i in range(0, N - B, B):
        batch, labels = X[i : i + B], Y[i : i + B]
        stats.collect_covs(batch, labels, B)

    stats.save_covs_sums()
    stats.load_covs_sums()

    cov_b, cov_w = stats.compute_covs()
    assert cov_b is None, cov_w is None

    stats.collect_covs(X[-B:], Y[-B:], B)
    cov_b, cov_w = stats.compute_covs()

    gram = X.mT @ X / stats.N1
    lda_sum = stats.mean_G.T @ stats.mean_G + cov_b + cov_w
    if not pt.allclose(gram, lda_sum, rtol=1e-4, atol=1e-7):
        print("data is unusual: gram != LDA")
    del cov_b, cov_w, X, Y, gram, lda_sum

    inv_snr = stats.inv_snr()
    print("inverse SNR:", inv_snr)

    W = pt.randn(C, D, dtype=dtype)
    duality = stats.self_duality(W)
    print("self-duality:", duality)

    coherence = stats.coherence()
    print("coherence:", coherence)
