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


def ld_layout_diagnostics(block_sizes, bucket_size=32):
    """Return cheap size and padding diagnostics for an LD block layout."""
    bucket_size = int(bucket_size)
    if bucket_size < 1:
        raise ValueError("bucket_size must be at least 1")

    sizes = np.asarray([int(size) for size in block_sizes if int(size) > 0])
    if not sizes.size:
        return {
            "active_blocks": 0,
            "variants": 0,
            "size_min": 0,
            "size_median": 0.0,
            "size_p90": 0.0,
            "size_max": 0,
            "padding_memory_ratio": 1.0,
            "padding_cubic_ratio": 1.0,
        }

    padded = ((sizes + bucket_size - 1) // bucket_size) * bucket_size
    return {
        "active_blocks": int(sizes.size),
        "variants": int(sizes.sum()),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_p90": float(np.percentile(sizes, 90)),
        "size_max": int(sizes.max()),
        "padding_memory_ratio": float(
            np.sum(padded.astype(np.float64) ** 2) /
            np.sum(sizes.astype(np.float64) ** 2)
        ),
        "padding_cubic_ratio": float(
            np.sum(padded.astype(np.float64) ** 3) /
            np.sum(sizes.astype(np.float64) ** 3)
        ),
    }


def diagnose_ld_blocks(ld_blocks, block_sizes, bucket_size=32,
                       rank_rtol=1e-8, ld_eigenvalues=None):
    """Measure padding, definiteness and numerical rank of real LD blocks."""
    rank_rtol = float(rank_rtol)
    if not 0.0 <= rank_rtol < 1.0:
        raise ValueError("rank_rtol must be in [0, 1)")

    diagnostics = ld_layout_diagnostics(block_sizes, bucket_size)
    if (ld_eigenvalues is not None and
            len(ld_eigenvalues) != len(ld_blocks)):
        raise ValueError(
            "ld_eigenvalues and ld_blocks must have the same length"
        )
    ranks = []
    rank_fractions = []
    minimum_eigenvalues = []
    condition_estimates = []

    for block_index, (ld, size) in enumerate(zip(ld_blocks, block_sizes)):
        size = int(size)
        if not size:
            continue
        if np.shape(ld) != (size, size):
            raise ValueError(
                "LD block %d has shape %s; expected (%d, %d)" %
                (block_index, np.shape(ld), size, size)
            )
        if ld_eigenvalues is None:
            symmetric_ld = (
                np.asarray(ld, dtype=np.float64) +
                np.asarray(ld, dtype=np.float64).T
            ) * 0.5
            eigenvalues = np.linalg.eigvalsh(symmetric_ld)
        else:
            eigenvalues = np.asarray(
                ld_eigenvalues[block_index], dtype=np.float64
            )
        maximum = max(float(eigenvalues[-1]), 0.0)
        cutoff = rank_rtol * maximum
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        ranks.append(rank)
        rank_fractions.append(rank / float(size))
        minimum_eigenvalues.append(float(eigenvalues[0]))

        positive = eigenvalues[eigenvalues > cutoff]
        if positive.size:
            condition_estimates.append(float(maximum / positive[0]))

    diagnostics.update({
        "rank_rtol": rank_rtol,
        "rank_min": min(ranks, default=0),
        "rank_median": float(np.median(ranks)) if ranks else 0.0,
        "rank_max": max(ranks, default=0),
        "rank_fraction_median": (
            float(np.median(rank_fractions)) if rank_fractions else 0.0
        ),
        "rank_fraction_p90": (
            float(np.percentile(rank_fractions, 90))
            if rank_fractions else 0.0
        ),
        "minimum_eigenvalue": min(minimum_eigenvalues, default=0.0),
        "condition_median": (
            float(np.median(condition_estimates))
            if condition_estimates else float("inf")
        ),
    })
    return diagnostics


def format_ld_diagnostics(diagnostics):
    """Format diagnostics as stable, grep-friendly profile lines."""
    lines = [
        "[LD] %d active blocks, %d variants; size min/median/p90/max "
        "%d/%.1f/%.1f/%d" %
        (
            diagnostics["active_blocks"], diagnostics["variants"],
            diagnostics["size_min"], diagnostics["size_median"],
            diagnostics["size_p90"], diagnostics["size_max"],
        ),
        "[LD] padding memory %.3fx; estimated cubic work %.3fx" %
        (
            diagnostics["padding_memory_ratio"],
            diagnostics["padding_cubic_ratio"],
        ),
    ]
    if "rank_rtol" in diagnostics:
        lines.extend([
            "[LD] numerical rank at rtol %.1e min/median/max %d/%.1f/%d; "
            "median fraction %.3f" %
            (
                diagnostics["rank_rtol"], diagnostics["rank_min"],
                diagnostics["rank_median"], diagnostics["rank_max"],
                diagnostics["rank_fraction_median"],
            ),
            "[LD] minimum eigenvalue %.3e; median retained condition %.3e" %
            (
                diagnostics["minimum_eigenvalue"],
                diagnostics["condition_median"],
            ),
        ])
    return "\n".join(lines)


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


class CudaDirectBetaBackend(CudaBetaBackend):
    """Preallocated direct cuSOLVER/cuBLAS FP64 batched implementation."""

    name = "cuda-direct"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from cupy.cuda import cublas, device
            from cupy_backends.cuda.libs import cusolver
        except ImportError as exc:
            raise RuntimeError(
                "The direct CUDA beta backend requires CuPy's cuBLAS and "
                "cuSOLVER bindings"
            ) from exc

        cp = self._cp
        self._cublas = cublas
        self._cusolver = cusolver
        self._one = np.array(1.0, dtype=np.float64)
        self._potrf_checked = False

        def matrix_pointers(array):
            count = array.shape[0]
            step = int(array[0].nbytes)
            start = int(array.data.ptr)
            return cp.arange(
                start,
                start + step * count,
                step,
                dtype=cp.uintp,
            )

        with self._device:
            self._cublas_handle = device.get_cublas_handle()
            self._cusolver_handle = device.get_cusolver_handle()
            for group in self._groups:
                precision = group["ld"].copy()
                rhs = cp.empty_like(group["beta_mrg"])
                direct_arrays = {
                    "precision": precision,
                    "rhs": rhs,
                    "precision_ptrs": matrix_pointers(precision),
                    "rhs_ptrs": matrix_pointers(rhs),
                    "potrf_info": cp.empty(
                        precision.shape[0], dtype=cp.int32
                    ),
                }
                group.update(direct_arrays)
                self._resident_bytes += sum(
                    int(value.nbytes) for value in direct_arrays.values()
                )

    def _triangular_solve(self, group, trans):
        size = group["precision"].shape[-1]
        count = group["precision"].shape[0]
        self._cublas.dtrsmBatched(
            self._cublas_handle,
            self._cublas.CUBLAS_SIDE_LEFT,
            self._cublas.CUBLAS_FILL_MODE_UPPER,
            trans,
            self._cublas.CUBLAS_DIAG_NON_UNIT,
            size,
            1,
            self._one.ctypes.data,
            group["precision_ptrs"].data.ptr,
            size,
            group["rhs_ptrs"].data.ptr,
            size,
            count,
        )

    def sample(self, psi, sigma):
        """Draw beta with fixed device workspaces and in-place CUDA calls."""
        cp = self._cp
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._p:
            raise ValueError("psi and beta_mrg must have the same length")

        with self._device:
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=cp.float64)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            noise = self._rng.standard_normal(self._p, dtype=cp.float64)

            for group_index, group in enumerate(self._groups):
                precision = group["precision"]
                cp.copyto(precision, group["ld"])
                safe_indices = group["indices"]
                inv_psi = cp.where(
                    group["valid"],
                    1.0 / self._psi_device[safe_indices],
                    0.0,
                )
                diag = group["diag"]
                precision[:, diag, diag] += inv_psi

                size = precision.shape[-1]
                count = precision.shape[0]
                self._cusolver.dpotrfBatched(
                    self._cusolver_handle,
                    self._cublas.CUBLAS_FILL_MODE_UPPER,
                    size,
                    group["precision_ptrs"].data.ptr,
                    size,
                    group["potrf_info"].data.ptr,
                    count,
                )
                if not self._potrf_checked:
                    info = cp.asnumpy(group["potrf_info"])
                    failures = np.flatnonzero(info)
                    if failures.size:
                        first = int(failures[0])
                        raise RuntimeError(
                            "direct CUDA Cholesky failed in size bucket %d, "
                            "matrix %d with info=%d" %
                            (group_index, first, int(info[first]))
                        )

                rhs = group["rhs"]
                cp.copyto(rhs, group["beta_mrg"])
                # Row-major L is seen by cuBLAS as column-major U=L.T.
                # U.T y=b therefore performs the first solve L y=b.
                self._triangular_solve(
                    group, self._cublas.CUBLAS_OP_T
                )
                rhs += (
                    sd * noise[group["indices"]][..., None] *
                    group["valid"][..., None]
                )
                quad += cp.sum(rhs * rhs)

                # U beta=y is the second solve L.T beta=y.
                self._triangular_solve(
                    group, self._cublas.CUBLAS_OP_N
                )
                flat_valid = group["valid"].ravel()
                self._beta_result[
                    group["indices"].ravel()[flat_valid]
                ] = rhs[..., 0].ravel()[flat_valid]

            self._potrf_checked = True
            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            "cuda-direct:%d (preallocated FP64 potrfBatched + "
            "trsmBatched; %d size buckets; %.1f MiB static resident)" %
            (
                self._device.id,
                len(self._groups),
                self._resident_bytes / (1024.0 * 1024.0),
            )
        )


class CudaPcgBetaBackend:
    """FP64 perturb-and-solve sampler using batched CUDA PCG solves."""

    name = "cuda-pcg"

    def __init__(self, ld_blocks, block_sizes, beta_mrg, n_gwas,
                 seed=None, cuda_device=0, cuda_bucket_size=32,
                 pcg_tol=1e-10, pcg_maxiter=100, pcg_check_interval=4,
                 ld_rank_tol=1e-8, ld_factors=None, ld_eigenvalues=None,
                 **_kwargs):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "The CUDA PCG backend requires CuPy. Install the package "
                "matching the host CUDA runtime (for example, "
                "cupy-cuda12x)."
            ) from exc

        bucket_size = int(cuda_bucket_size)
        if bucket_size < 1:
            raise ValueError("cuda_bucket_size must be at least 1")
        pcg_tol = float(pcg_tol)
        if not 0.0 < pcg_tol < 1.0:
            raise ValueError("pcg_tol must be between 0 and 1")
        pcg_maxiter = int(pcg_maxiter)
        if pcg_maxiter < 1:
            raise ValueError("pcg_maxiter must be at least 1")
        pcg_check_interval = int(pcg_check_interval)
        if pcg_check_interval < 1:
            raise ValueError("pcg_check_interval must be at least 1")
        ld_rank_tol = float(ld_rank_tol)
        if not 0.0 <= ld_rank_tol < 1.0:
            raise ValueError("ld_rank_tol must be in [0, 1)")

        self._cp = cp
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
        self._pcg_tol = pcg_tol
        self._pcg_maxiter = pcg_maxiter
        self._pcg_check_interval = pcg_check_interval
        self._ld_rank_tol = ld_rank_tol
        self._solve_groups = 0
        self._iteration_steps = 0
        self._maximum_steps = 0
        self._maximum_residual = 0.0

        beta_mrg = np.asarray(beta_mrg, dtype=np.float64).reshape(-1)
        self._p = beta_mrg.size
        layout = _block_layout(ld_blocks, block_sizes, self._p)
        if ld_factors is not None and len(ld_factors) != len(ld_blocks):
            raise ValueError("ld_factors and ld_blocks must have the same length")
        if (ld_eigenvalues is not None and
                len(ld_eigenvalues) != len(ld_blocks)):
            raise ValueError(
                "ld_eigenvalues and ld_blocks must have the same length"
            )

        prepared = {}
        effective_ranks = []
        for block_index, block_slice in layout:
            size = block_slice.stop - block_slice.start
            ld = np.asarray(ld_blocks[block_index], dtype=np.float64)
            if ld_factors is None:
                symmetric_ld = (ld + ld.T) * 0.5
                eigenvalues, eigenvectors = np.linalg.eigh(symmetric_ld)
                scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
                negative_tolerance = (
                    100.0 * np.finfo(np.float64).eps * size * scale
                )
                if float(eigenvalues[0]) < -negative_tolerance:
                    raise ValueError(
                        "LD block %d is not positive semidefinite: minimum "
                        "eigenvalue %.3e" %
                        (block_index, float(eigenvalues[0]))
                    )
                eigenvalues = np.maximum(eigenvalues, 0.0)
                factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
                # Use the same clipped matrix for the solve and perturbation.
                ld = np.dot(
                    eigenvectors * eigenvalues[None, :], eigenvectors.T
                )
            else:
                factor = np.asarray(
                    ld_factors[block_index], dtype=np.float64
                )
                if factor.shape[0] != size:
                    raise ValueError(
                        "LD factor %d has %d rows; expected %d" %
                        (block_index, factor.shape[0], size)
                    )
                if ld_eigenvalues is None:
                    eigenvalues = np.sum(factor * factor, axis=0)
                else:
                    eigenvalues = np.asarray(
                        ld_eigenvalues[block_index], dtype=np.float64
                    )

            maximum = (
                max(float(np.max(eigenvalues)), 0.0)
                if eigenvalues.size else 0.0
            )
            cutoff = ld_rank_tol * maximum
            effective_ranks.append(
                int(np.count_nonzero(eigenvalues > cutoff))
            )
            if eigenvalues.size != factor.shape[1]:
                raise ValueError(
                    "LD factor %d has %d columns, but %d eigenvalues" %
                    (block_index, factor.shape[1], eigenvalues.size)
                )
            # Removing exactly zero PSD components is exact, unlike the
            # rank_rtol diagnostic threshold, which never truncates a draw.
            factor = factor[:, eigenvalues > 0.0]
            prepared[block_index] = (ld, factor)

        buckets = {}
        for block_index, block_slice in layout:
            size = block_slice.stop - block_slice.start
            padded_size = ((size + bucket_size - 1) // bucket_size) * bucket_size
            rank = prepared[block_index][1].shape[1]
            padded_rank = (
                ((rank + bucket_size - 1) // bucket_size) * bucket_size
                if rank else 0
            )
            buckets.setdefault((padded_size, padded_rank), []).append(
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

            for (padded_size, padded_rank), blocks in sorted(buckets.items()):
                count = len(blocks)
                host_ld = np.zeros(
                    (count, padded_size, padded_size), dtype=np.float64
                )
                host_sqrt_ld = np.zeros(
                    (count, padded_size, padded_rank), dtype=np.float64
                )
                host_beta_mrg = np.zeros(
                    (count, padded_size), dtype=np.float64
                )
                host_indices = np.zeros(
                    (count, padded_size), dtype=np.int64
                )
                host_valid = np.zeros(
                    (count, padded_size), dtype=np.bool_
                )

                for row, (block_index, block_slice) in enumerate(blocks):
                    size = block_slice.stop - block_slice.start
                    ld, factor = prepared[block_index]
                    rank = factor.shape[1]
                    host_ld[row, :size, :size] = ld
                    host_sqrt_ld[row, :size, :rank] = factor
                    host_beta_mrg[row, :size] = beta_mrg[block_slice]
                    host_indices[row, :size] = np.arange(
                        block_slice.start, block_slice.stop
                    )
                    host_valid[row, :size] = True

                group = {
                    "ld": cp.asarray(host_ld, blocking=True),
                    "sqrt_ld": cp.asarray(host_sqrt_ld, blocking=True),
                    "beta_mrg": cp.asarray(host_beta_mrg, blocking=True),
                    "indices": cp.asarray(host_indices, blocking=True),
                    "valid": cp.asarray(host_valid, blocking=True),
                    "ld_diag": cp.asarray(
                        np.diagonal(host_ld, axis1=1, axis2=2).copy(),
                        blocking=True,
                    ),
                    "factor_rank": padded_rank,
                }
                self._resident_bytes += sum(
                    int(value.nbytes) for value in group.values()
                    if hasattr(value, "nbytes")
                )
                self._groups.append(group)

        layout_diagnostics = ld_layout_diagnostics(
            block_sizes, bucket_size
        )
        self._padding_cubic_ratio = layout_diagnostics[
            "padding_cubic_ratio"
        ]
        self._rank_median = (
            float(np.median(effective_ranks)) if effective_ranks else 0.0
        )
        self._rank_max = max(effective_ranks, default=0)

    def _apply_precision(self, group, diagonal, value):
        cp = self._cp
        return (
            cp.matmul(group["ld"], value[..., None])[..., 0] +
            diagonal * value
        )

    def _solve(self, group, diagonal, rhs):
        """Solve a group of SPD systems with masked batched FP64 PCG."""
        cp = self._cp
        solution = cp.zeros_like(rhs)
        residual = rhs.copy()
        preconditioned = residual / (group["ld_diag"] + diagonal)
        direction = preconditioned.copy()
        rz = cp.sum(residual * preconditioned, axis=1)
        rhs_norm = cp.linalg.norm(rhs, axis=1)
        # Leave headroom for drift between recursive and true residuals.
        target = (self._pcg_tol * 0.25) * cp.maximum(
            rhs_norm, cp.finfo(cp.float64).tiny
        )
        residual_norm = cp.linalg.norm(residual, axis=1)
        active = residual_norm > target

        steps = 0
        for steps in range(1, self._pcg_maxiter + 1):
            precision_direction = self._apply_precision(
                group, diagonal, direction
            )
            denominator = cp.sum(
                direction * precision_direction, axis=1
            )
            safe = active & (denominator > 0.0)
            alpha = cp.where(
                safe, rz / cp.where(safe, denominator, 1.0), 0.0
            )
            solution += alpha[:, None] * direction
            residual -= alpha[:, None] * precision_direction

            residual_norm = cp.linalg.norm(residual, axis=1)
            active = residual_norm > target
            preconditioned = (
                residual / (group["ld_diag"] + diagonal)
            ) * active[:, None]
            rz_new = cp.sum(residual * preconditioned, axis=1)
            update = active & (rz != 0.0)
            coefficient = cp.where(
                update, rz_new / cp.where(update, rz, 1.0), 0.0
            )
            direction = (
                preconditioned + coefficient[:, None] * direction
            ) * active[:, None]
            rz = rz_new

            if (steps % self._pcg_check_interval == 0 and
                    not bool(cp.any(active).item())):
                break

        # Recompute the true residual once. Recursive PCG residuals can drift.
        precision_solution = self._apply_precision(
            group, diagonal, solution
        )
        true_residual = rhs - precision_solution
        residual_ratio = cp.linalg.norm(true_residual, axis=1) / cp.maximum(
            rhs_norm, cp.finfo(cp.float64).tiny
        )
        maximum_residual = float(cp.max(residual_ratio).item())
        if maximum_residual > self._pcg_tol:
            failed = int(cp.count_nonzero(
                residual_ratio > self._pcg_tol
            ).item())
            raise RuntimeError(
                "CUDA PCG failed to converge for %d block(s) in %d "
                "iterations: maximum relative residual %.3e exceeds %.3e" %
                (failed, self._pcg_maxiter, maximum_residual,
                 self._pcg_tol)
            )

        self._solve_groups += 1
        self._iteration_steps += steps
        self._maximum_steps = max(self._maximum_steps, steps)
        self._maximum_residual = max(
            self._maximum_residual, maximum_residual
        )
        return solution, precision_solution

    def sample(self, psi, sigma):
        """Draw beta via perturb-and-solve and return (beta, quadratic)."""
        cp = self._cp
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        if psi.size != self._p:
            raise ValueError("psi and beta_mrg must have the same length")

        with self._device:
            self._psi_device.set(psi)
            quad = cp.zeros((), dtype=cp.float64)
            sd = float(np.sqrt(float(sigma) / self._n_gwas))
            diagonal_noise = self._rng.standard_normal(
                self._p, dtype=cp.float64
            )

            for group in self._groups:
                indices = group["indices"]
                valid = group["valid"]
                diagonal = cp.where(
                    valid, 1.0 / self._psi_device[indices], 1.0
                )
                z_diagonal = diagonal_noise[indices] * valid
                z_ld = self._rng.standard_normal(
                    (indices.shape[0], group["factor_rank"]),
                    dtype=cp.float64,
                )
                if group["factor_rank"]:
                    ld_perturbation = cp.matmul(
                        group["sqrt_ld"], z_ld[..., None]
                    )[:, :, 0]
                else:
                    ld_perturbation = cp.zeros_like(z_diagonal)
                perturbation = (
                    cp.sqrt(diagonal) * z_diagonal +
                    ld_perturbation
                )
                rhs = (
                    group["beta_mrg"] + sd * perturbation
                ) * valid
                beta_batch, precision_beta = self._solve(
                    group, diagonal, rhs
                )

                flat_valid = valid.ravel()
                self._beta_result[
                    indices.ravel()[flat_valid]
                ] = beta_batch.ravel()[flat_valid]
                quad += cp.sum(beta_batch * precision_beta)

            self._beta_result[self._p] = quad
            result = cp.asnumpy(self._beta_result)

        return result[:self._p].reshape(-1, 1), float(result[self._p])

    def describe(self):
        return (
            "cuda-pcg:%d (FP64 perturb-and-solve; %d size/rank buckets; "
            "rank median/max %.1f/%d at %.1e; padding %.3fx; %.1f MiB "
            "static resident; tol %.1e, maxiter %d)" %
            (
                self._device.id, len(self._groups), self._rank_median,
                self._rank_max, self._ld_rank_tol,
                self._padding_cubic_ratio,
                self._resident_bytes / (1024.0 * 1024.0),
                self._pcg_tol, self._pcg_maxiter,
            )
        )

    def profile_summary(self):
        if not self._solve_groups:
            return "CUDA PCG: no solves recorded"
        return (
            "CUDA PCG: mean %.1f iterations/group, max %d, maximum "
            "true relative residual %.3e" %
            (
                self._iteration_steps / float(self._solve_groups),
                self._maximum_steps, self._maximum_residual,
            )
        )


def make_beta_backend(backend, ld_blocks, block_sizes, beta_mrg, n_gwas,
                      seed=None, cuda_device=0, cuda_bucket_size=32,
                      pcg_tol=1e-10, pcg_maxiter=100,
                      pcg_check_interval=4, ld_rank_tol=1e-8,
                      ld_factors=None, ld_eigenvalues=None):
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
        "pcg_tol": pcg_tol,
        "pcg_maxiter": pcg_maxiter,
        "pcg_check_interval": pcg_check_interval,
        "ld_rank_tol": ld_rank_tol,
        "ld_factors": ld_factors,
        "ld_eigenvalues": ld_eigenvalues,
    }
    if backend == "cpu":
        return CpuBetaBackend(**kwargs)
    if backend == "cuda":
        return CudaBetaBackend(**kwargs)
    if backend == "cuda-direct":
        return CudaDirectBetaBackend(**kwargs)
    if backend == "cuda-pcg":
        return CudaPcgBetaBackend(**kwargs)
    raise ValueError(
        "unknown beta backend %r; expected 'cpu', 'cuda', 'cuda-direct' "
        "or 'cuda-pcg'" % backend
    )
