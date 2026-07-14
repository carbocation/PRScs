#!/usr/bin/env python3

"""CPU and CUDA implementations of the PRS-CS beta block update."""


import numpy as np
from scipy import linalg


def _block_layout(ld_blocks, block_sizes, p):
    """Validate block inputs and return (block_index, slice) pairs."""
    if len(ld_blocks) != len(block_sizes):
        raise ValueError("ld_blocks and block_sizes must have the same length")

    sizes = [int(size) for size in block_sizes]
    if any(size < 0 for size in sizes):
        raise ValueError("block sizes must be non-negative")
    if sum(sizes) != p:
        raise ValueError(
            "block sizes cover %d variants, but beta_mrg contains %d" %
            (sum(sizes), p)
        )

    layout = []
    start = 0
    for block_index, (ld, size) in enumerate(zip(ld_blocks, sizes)):
        if size:
            if np.shape(ld) != (size, size):
                raise ValueError(
                    "LD block %d has shape %s; expected (%d, %d)" %
                    (block_index, np.shape(ld), size, size)
                )
            layout.append((block_index, slice(start, start + size)))
        start += size
    return layout


class CpuBetaBackend:
    """Optimized SciPy implementation used as the correctness baseline."""

    name = "cpu"

    def __init__(self, ld_blocks, block_sizes, beta_mrg, n_gwas, **_kwargs):
        self._ld_blocks = ld_blocks
        self._beta_mrg = np.asarray(beta_mrg, dtype=np.float64).reshape(-1, 1)
        self._n_gwas = int(n_gwas)
        self._layout = _block_layout(
            ld_blocks, block_sizes, self._beta_mrg.shape[0]
        )
        self._beta = np.empty_like(self._beta_mrg)
        self._work = {
            block_index: np.empty(np.shape(ld_blocks[block_index]),
                                  dtype=np.float64, order="F")
            for block_index, _ in self._layout
        }

    def sample(self, psi, sigma):
        """Draw beta for every non-empty LD block and return (beta, quad)."""
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._beta_mrg.shape[0]:
            raise ValueError("psi and beta_mrg must have the same length")

        sd = float(np.sqrt(float(sigma) / self._n_gwas))
        quad = 0.0

        for block_index, block_slice in self._layout:
            size = block_slice.stop - block_slice.start
            precision = self._work[block_index]
            np.copyto(precision, self._ld_blocks[block_index])
            diag = np.diag_indices(size)
            precision[diag] += 1.0 / psi[block_slice]

            # SciPy returns U with A = U.T @ U when lower=False.
            chol = linalg.cholesky(
                precision,
                lower=False,
                overwrite_a=True,
                check_finite=False,
            )
            beta_tmp = linalg.solve_triangular(
                chol,
                self._beta_mrg[block_slice],
                trans="T",
                lower=False,
                check_finite=False,
            )
            beta_tmp += sd * np.random.standard_normal((size, 1))
            self._beta[block_slice] = linalg.solve_triangular(
                chol,
                beta_tmp,
                trans="N",
                lower=False,
                check_finite=False,
            )

            # A = U.T @ U and U @ beta = beta_tmp, hence
            # beta.T @ A @ beta = ||beta_tmp||^2. This avoids a dense
            # matrix-vector product that the original implementation did.
            quad += float(np.dot(beta_tmp[:, 0], beta_tmp[:, 0]))

        return self._beta, quad

    def describe(self):
        return "cpu (SciPy Cholesky; %d active LD blocks)" % len(self._layout)


class CudaBetaBackend:
    """LD-resident, batched CuPy implementation of the beta block update."""

    name = "cuda"

    def __init__(self, ld_blocks, block_sizes, beta_mrg, n_gwas,
                 seed=None, cuda_device=0, cuda_bucket_size=32, **_kwargs):
        try:
            import cupy as cp
            from cupyx.scipy.linalg import solve_triangular
        except ImportError as exc:
            raise RuntimeError(
                "The CUDA backend requires CuPy 14.1 or newer. Install the "
                "package matching the host CUDA runtime (for example, "
                "cupy-cuda12x)."
            ) from exc

        version = tuple(
            int(part) for part in cp.__version__.split(".")[:2]
            if part.isdigit()
        )
        if version and version < (14, 1):
            raise RuntimeError(
                "The CUDA backend requires CuPy 14.1 or newer for batched "
                "triangular solves; found CuPy %s." % cp.__version__
            )

        bucket_size = int(cuda_bucket_size)
        if bucket_size < 1:
            raise ValueError("cuda_bucket_size must be at least 1")

        self._cp = cp
        self._solve_triangular = solve_triangular
        device_id = int(cuda_device)
        if device_id < 0:
            raise ValueError("cuda_device must be non-negative")
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise RuntimeError(
                "CuPy is installed, but the CUDA runtime is unavailable"
            ) from exc
        if device_count < 1:
            raise RuntimeError("CuPy is installed, but no CUDA device is visible")
        if device_id >= device_count:
            raise ValueError(
                "cuda_device %d was requested, but only %d CUDA device(s) "
                "are visible" % (device_id, device_count)
            )

        self._device = cp.cuda.Device(device_id)
        self._n_gwas = int(n_gwas)
        self._bucket_size = bucket_size

        beta_mrg = np.asarray(beta_mrg, dtype=np.float64).reshape(-1, 1)
        self._p = beta_mrg.shape[0]
        layout = _block_layout(ld_blocks, block_sizes, self._p)

        buckets = {}
        for block_index, block_slice in layout:
            size = block_slice.stop - block_slice.start
            padded_size = ((size + bucket_size - 1) // bucket_size) * bucket_size
            buckets.setdefault(padded_size, []).append(
                (block_index, block_slice)
            )

        self._groups = []
        self._resident_bytes = 0
        with self._device:
            self._rng = cp.random.RandomState(seed)
            self._beta_result = cp.empty(self._p + 1, dtype=cp.float64)
            self._psi_device = cp.empty(self._p, dtype=cp.float64)
            self._resident_bytes += (
                int(self._beta_result.nbytes) + int(self._psi_device.nbytes)
            )

            for padded_size, blocks in sorted(buckets.items()):
                count = len(blocks)
                host_ld = np.zeros(
                    (count, padded_size, padded_size), dtype=np.float64
                )
                host_beta_mrg = np.zeros(
                    (count, padded_size, 1), dtype=np.float64
                )
                host_indices = np.zeros(
                    (count, padded_size), dtype=np.int64
                )
                host_valid = np.zeros(
                    (count, padded_size), dtype=np.bool_
                )

                for row, (block_index, block_slice) in enumerate(blocks):
                    size = block_slice.stop - block_slice.start
                    host_ld[row, :size, :size] = ld_blocks[block_index]
                    if size < padded_size:
                        padding = np.arange(size, padded_size)
                        host_ld[row, padding, padding] = 1.0
                    host_beta_mrg[row, :size, 0] = beta_mrg[block_slice, 0]
                    host_indices[row, :size] = np.arange(
                        block_slice.start, block_slice.stop
                    )
                    host_valid[row, :size] = True

                group = {
                    "ld": cp.asarray(host_ld, blocking=True),
                    "beta_mrg": cp.asarray(host_beta_mrg, blocking=True),
                    "indices": cp.asarray(host_indices, blocking=True),
                    "valid": cp.asarray(host_valid, blocking=True),
                    "diag": cp.arange(padded_size),
                }
                self._resident_bytes += sum(
                    int(value.nbytes) for value in group.values()
                )
                self._groups.append(group)

    def sample(self, psi, sigma):
        """Draw beta on CUDA, copying only O(p) state per iteration."""
        cp = self._cp
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._p:
            raise ValueError("psi and beta_mrg must have the same length")

        with self._device:
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=cp.float64)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(self._p, dtype=cp.float64)

            for group in self._groups:
                precision = group["ld"].copy()
                safe_indices = group["indices"]
                inv_psi = cp.where(
                    group["valid"],
                    1.0 / self._psi_device[safe_indices],
                    0.0,
                )
                diag = group["diag"]
                precision[:, diag, diag] += inv_psi

                # CuPy returns L with A = L @ L.T. Both linalg routines
                # operate on the entire batch, including padded dimensions.
                chol = cp.linalg.cholesky(precision)
                beta_tmp = self._solve_triangular(
                    chol,
                    group["beta_mrg"],
                    trans="N",
                    lower=True,
                    check_finite=False,
                )
                beta_tmp += (
                    sd * noise[group["indices"]][..., None] *
                    group["valid"][..., None]
                )
                beta_batch = self._solve_triangular(
                    chol,
                    beta_tmp,
                    trans="T",
                    lower=True,
                    check_finite=False,
                )

                flat_valid = group["valid"].ravel()
                self._beta_result[
                    group["indices"].ravel()[flat_valid]
                ] = beta_batch[..., 0].ravel()[flat_valid]
                quad += cp.sum(beta_tmp * beta_tmp)

            # One device-to-host copy and synchronization per MCMC iteration.
            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            "cuda:%d (CuPy batched Cholesky; %d size buckets; %.1f MiB "
            "static resident)" %
            (self._device.id, len(self._groups),
             self._resident_bytes / (1024.0 * 1024.0))
        )


def make_beta_backend(backend, ld_blocks, block_sizes, beta_mrg, n_gwas,
                      seed=None, cuda_device=0, cuda_bucket_size=32):
    """Construct a beta sampler without importing CUDA dependencies on CPU."""
    backend = str(backend).lower()
    kwargs = {
        "ld_blocks": ld_blocks,
        "block_sizes": block_sizes,
        "beta_mrg": beta_mrg,
        "n_gwas": n_gwas,
        "seed": seed,
        "cuda_device": cuda_device,
        "cuda_bucket_size": cuda_bucket_size,
    }
    if backend == "cpu":
        return CpuBetaBackend(**kwargs)
    if backend == "cuda":
        return CudaBetaBackend(**kwargs)
    raise ValueError("unknown beta backend %r; expected 'cpu' or 'cuda'" % backend)
