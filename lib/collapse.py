import os
from typing import Tuple, Union

import torch as pt
from neural_collapse.accumulate import (DecAccumulator, MeanAccumulator,
                                        VarNormAccumulator)
from neural_collapse.util import hashify
from torch import Tensor


class CollapseStatistics:
    def __init__(
        self,
        C: int = None,
        D: int = None,
        device: Union[str, pt.device] = "cpu",
        dtype: pt.dtype = float,
        means_path: str = None,
        vars_path: str = None,
        decs_path: str = None,
        verbose: bool = True,
    ):
        self.device = device
        self.dtype, self.eps = dtype, pt.finfo(dtype).eps
        self.means_hash = None  # ensuring covariances based on consistent means

        if means_path:
            self.load_mean_totals(means_path, verbose)
            assert type(self.means_accum) == MeanAccumulator
        else:
            assert type(C) == type(D) == int
            self.C, self.D = C, D
            self.means_accum = MeanAccumulator(C, D, device, dtype)
            self.N_seqs_means = 0

        self.vars_accum = VarNormAccumulator(self.C, self.D, device, dtype)
        if vars_path:
            self.load_vars_totals(vars_path, verbose)
        else:
            self.N_seqs_vars = 0

        self.decs_accum = DecAccumulator(self.C, self.D, device)
        if decs_path:
            self.load_decs_totals(decs_path)
        else:
            self.N_seqs_decs = 0

    # methods for collecting/saving/loading means
    def collect_mean_totals(
        self, X: Tensor, Y: Tensor, n_seqs: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """First pass: increment vector counts and totals for a batch.
        B: batch size
        X (B x D): feature vectors
        Y (B x 1): class labels
        """
        if len(Y.shape) < 1:
            print("WARN: batch too short")
            return None, None
        self.N_seqs_means += n_seqs

        return self.means_accum.accumulate(X, Y)

    def save_mean_totals(self, file: str, verbose: bool = False) -> str:
        """Save totals (used with counts for computing means).
        file: path to save totals data.
        verbose: logging flag.
        """
        self.means_hash = hashify(self.means_accum.compute()[0])

        data = {
            "hash": self.means_hash,
            "N_seqs": self.N_seqs_means,
            "Ns_toks": self.means_accum.ns_samples,
            "totals": self.means_accum.totals,
        }
        pt.save(data, file)

        if verbose:
            N_seqs, N_toks = self.N_seqs_means, self.means_accum.ns_samples.sum().item()
            print(f"SAVED means to {file}; {N_toks} in {N_seqs} seqs")
            print(f"  HASH: {self.means_hash}")
        return file

    def load_mean_totals(self, file: str, verbose: bool = False) -> int:
        """Load totals (used with counts for computing means).
        file: path to load totals data.
        verbose: logging flag.
        """
        if not os.path.isfile(file):
            print(f"  W: file {file} not found; need to collect means from scratch")
            return 0
        data = pt.load(file, self.device, weights_only=True)

        assert self.means_hash in [None, data["hash"]], "overwriting current data"
        self.means_hash, self.N_seqs_means = data["hash"], data["N_seqs"]
        Ns_toks = data["Ns_toks"].to(self.device).squeeze()
        totals = data["totals"].to(self.device)

        self.C, self.D = totals.shape
        assert Ns_toks.shape[0] == self.C
        self.means_accum = MeanAccumulator(self.C, self.D, self.device, self.dtype)
        self.means_accum.ns_samples, self.means_accum.totals = Ns_toks, totals

        if verbose:
            N_toks = Ns_toks.sum().item()
            print(f"LOADED means from {file}; {N_toks} in {self.N_seqs_means} seqs")
        return self.N_seqs_means

    # methods for collecting/saving/loading variance norms
    def collect_vars_totals(self, X: Tensor, Y: Tensor, n_seqs: int = 1) -> Tensor:
        """Second pass: increment within/between-class covariance for a batch.
        B: batch size
        X (B x D): feature vectors
        Y (B x 1): class labels
        """
        if len(Y.shape) < 1:
            print("WARN: batch too short")
            return None, None
        N_means = self.means_accum.ns_samples.sum().item()
        N_vars = self.vars_accum.ns_samples.sum().item()
        if N_vars + Y.shape[0] > N_means:
            print("  W: this vars batch would exceed means samples")
            print(f"  {N_vars}+{Y.shape[0]} > {N_means}")
            return None, None
        self.N_seqs_vars += n_seqs

        M = self.means_accum.compute()[0]
        return self.vars_accum.accumulate(X, Y, M)

    def save_vars_totals(self, file: str, verbose: bool = False) -> str:
        """Save variance norm totals (used with counts for normalized variances).
        file: path to save variance sums data.
        verbose: logging flag.
        """
        data = {
            "hash": self.means_hash,
            "N_seqs": self.N_seqs_vars,
            "Ns_toks": self.vars_accum.ns_samples,
            "totals": self.vars_accum.totals,
        }
        pt.save(data, file)

        if verbose:
            N_seqs, N_toks = self.N_seqs_vars, self.vars_accum.ns_samples.sum().item()
            print(f"SAVED vars to {file}; {N_toks} in {N_seqs} seqs")
            print(f"  MEANS HASH: {self.means_hash}")
        return file

    def load_vars_totals(self, file: str, verbose: bool = False) -> int:
        """Load variance sums (used with counts for normalized variances).
        file: path to load variance sums data.
        verbose: logging flag.
        """
        means_file = file.replace("vars", "means")
        if not self.load_mean_totals(means_file, False):
            print("  W: means not found; please collect them first")
            return 0
        self.C, self.D = self.means_accum.n_classes, self.means_accum.d_vectors

        if not os.path.isfile(file):
            print(f"  W: path {file} not found; need to collect vars from scratch")
            return 0

        data = pt.load(file, self.device, weights_only=True)
        assert self.means_hash in [
            None,
            data["hash"],
        ], f"vars based on outdated means: {self.means_hash[:6]} != {data['hash'][:6]}"
        self.vars_accum.hash_M = self.means_hash = data["hash"]

        self.N_seqs_vars = data["N_seqs"]
        self.vars_accum.ns_samples = data["Ns_toks"].to(self.device)
        self.vars_accum.totals = data["totals"].to(self.device)
        assert len(self.vars_accum.ns_samples) == len(self.vars_accum.totals) == self.C

        if verbose:
            N_toks = self.vars_accum.ns_samples.sum().item()
            print(f"LOADED vars from {file}; {N_toks} in {self.N_seqs_vars} seqs")
        return self.N_seqs_vars

    # methods for collecting/saving/loading decision agreements
    def collect_decs_hits(
        self, X: Tensor, Y: Tensor, W: Tensor, n_seqs: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """Third pass: increment samples where near-class and model classifiers agree.
        B: batch size
        X (B x D): feature vectors
        Y (B x 1): class labels
        W (C x D): model classifier weights
        """
        if len(Y.shape) < 1:
            print("WARN: batch too short")
            return None, None
        N_means = self.means_accum.ns_samples.sum().item()
        N_decs = self.decs_accum.ns_samples.sum().item()
        if N_decs + Y.shape[0] > N_means:
            print("  W: this decs batch would exceed means samples")
            print(f"  {N_decs}+{Y.shape[0]} > {N_means}")
            return None, None
        self.N_seqs_decs += n_seqs

        M = None
        if self.decs_accum.index is None:
            M = self.means_accum.compute()[0]
        return self.decs_accum.accumulate(X, Y, W, M)

    def save_decs_totals(self, file: str, verbose: bool = False) -> str:
        """Save decision matches/misses.
        file: path to save decision counts.
        verbose: logging flag.
        """
        data = {
            "hash": self.means_hash,
            "N_seqs": self.N_seqs_decs,
            "Ns_toks": self.decs_accum.ns_samples,
            "totals": self.decs_accum.totals,
        }
        pt.save(data, file)

        if verbose:
            N_seqs, N_toks = self.N_seqs_decs, self.decs_accum.ns_samples.sum().item()
            print(f"SAVED decs to {file}; {N_toks} in {N_seqs} seqs")
            print(f"  MEANS HASH: {self.means_hash}")
        return file

    def load_decs_totals(self, file: str, verbose: bool = False) -> int:
        """Load decision matches/misses.
        file: path to load decision counts.
        verbose: logging flag.
        """
        means_file = file.replace("decs", "means")
        if not self.load_mean_totals(means_file, False):
            print("  W: means not found; please collect them first")
            return 0
        self.C, self.D = self.means_accum.n_classes, self.means_accum.d_vectors

        if not os.path.isfile(file):
            print(f"  W: path {file} not found; need to collect decs from scratch")
            return 0

        data = pt.load(file, self.device, weights_only=True)
        assert self.means_hash in [
            None,
            data["hash"],
        ], f"decs based on outdated means: {self.means_hash[:6]} != {data['hash'][:6]}"
        self.decs_accum.hash_M = self.means_hash = data["hash"]

        self.N_seqs_decs = data["N_seqs"]
        self.decs_accum.ns_samples = data["Ns_toks"].to(self.device)
        self.decs_accum.totals = data["totals"].to(self.device)
        assert len(self.decs_accum.ns_samples) == len(self.decs_accum.totals) == self.C

        if verbose:
            N_toks = self.decs_accum.ns_samples.sum().item()
            print(f"LOADED decs from {file}; {N_toks} in {self.N_seqs_decs} seqs")
        return self.N_seqs_decs
