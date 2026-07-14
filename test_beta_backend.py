#!/usr/bin/env python3

"""Correctness tests for the CPU reference and optional CUDA beta backend."""


import unittest

import numpy as np
from scipy import linalg

from beta_backend import CpuBetaBackend, CudaBetaBackend, make_beta_backend


def _inputs():
    rng = np.random.default_rng(7)
    sizes = [3, 0, 5, 7]
    blocks = []
    for size in sizes:
        if not size:
            blocks.append(np.array([]))
            continue
        x = rng.normal(size=(size, size))
        blocks.append(x @ x.T / size + np.eye(size) * 0.2)

    p = sum(sizes)
    beta_mrg = rng.normal(size=(p, 1))
    psi = rng.uniform(0.1, 1.0, size=(p, 1))
    return blocks, sizes, beta_mrg, psi


def _legacy_sample(blocks, sizes, beta_mrg, psi, sigma, n_gwas):
    beta = np.zeros_like(beta_mrg)
    quad = 0.0
    start = 0
    for ld, size in zip(blocks, sizes):
        if not size:
            continue
        block_slice = slice(start, start + size)
        precision = ld + np.diag(1.0 / psi[block_slice, 0])
        chol = linalg.cholesky(precision)
        beta_tmp = linalg.solve_triangular(
            chol, beta_mrg[block_slice], trans="T"
        )
        beta_tmp += np.sqrt(sigma / n_gwas) * np.random.standard_normal(
            (size, 1)
        )
        beta[block_slice] = linalg.solve_triangular(chol, beta_tmp)
        quad += (
            beta[block_slice].T @ precision @ beta[block_slice]
        ).item()
        start += size
    return beta, quad


def _cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CpuBetaBackendTests(unittest.TestCase):
    def test_matches_original_block_update(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        n_gwas = 1000

        np.random.seed(123)
        expected_beta, expected_quad = _legacy_sample(
            blocks, sizes, beta_mrg, psi, sigma, n_gwas
        )
        np.random.seed(123)
        actual_beta, actual_quad = CpuBetaBackend(
            blocks, sizes, beta_mrg, n_gwas
        ).sample(psi, sigma)

        np.testing.assert_allclose(actual_beta, expected_beta, rtol=1e-12,
                                   atol=1e-12)
        self.assertAlmostEqual(actual_quad, expected_quad, places=12)

    def test_rejects_inconsistent_layout(self):
        blocks, sizes, beta_mrg, _ = _inputs()
        sizes = list(sizes)
        sizes[-1] -= 1
        with self.assertRaisesRegex(ValueError, "block sizes cover"):
            CpuBetaBackend(blocks, sizes, beta_mrg, 1000)

    def test_factory_rejects_unknown_backend(self):
        blocks, sizes, beta_mrg, _ = _inputs()
        with self.assertRaisesRegex(ValueError, "unknown beta backend"):
            make_beta_backend("quantum", blocks, sizes, beta_mrg, 1000)


@unittest.skipUnless(_cuda_available(), "CuPy and a CUDA device are required")
class CudaBetaBackendTests(unittest.TestCase):
    def test_irregular_padded_blocks_have_correct_quadratic_form(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        backend = CudaBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        beta, quad = backend.sample(psi, sigma)

        expected_quad = 0.0
        start = 0
        for ld, size in zip(blocks, sizes):
            if not size:
                continue
            block_slice = slice(start, start + size)
            precision = ld + np.diag(1.0 / psi[block_slice, 0])
            expected_quad += (
                beta[block_slice].T @ precision @ beta[block_slice]
            ).item()
            start += size

        self.assertTrue(np.isfinite(beta).all())
        self.assertAlmostEqual(quad, expected_quad, places=9)


if __name__ == "__main__":
    unittest.main()
