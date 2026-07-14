#!/usr/bin/env python3

"""Correctness tests for the CPU reference and optional CUDA beta backend."""


import os
import tempfile
import unittest

import h5py
import numpy as np
from scipy import linalg

from beta_backend import (
    CpuBetaBackend,
    CudaBetaBackend,
    CudaDirectBetaBackend,
    CudaFp32BetaBackend,
    CudaHybridBetaBackend,
    CudaPcgBetaBackend,
    CudaStreamsBetaBackend,
    diagnose_ld_blocks,
    format_ld_diagnostics,
    ld_layout_diagnostics,
    make_beta_backend,
)
from parse_genet import _project_ld_psd, parse_ldblk


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

    def test_layout_diagnostics_measure_padding_cost(self):
        diagnostics = ld_layout_diagnostics([3, 0, 5, 7], bucket_size=4)
        self.assertEqual(diagnostics["active_blocks"], 3)
        self.assertEqual(diagnostics["variants"], 15)
        self.assertAlmostEqual(
            diagnostics["padding_memory_ratio"],
            (4**2 + 8**2 + 8**2) / float(3**2 + 5**2 + 7**2),
        )
        self.assertAlmostEqual(
            diagnostics["padding_cubic_ratio"],
            (4**3 + 8**3 + 8**3) / float(3**3 + 5**3 + 7**3),
        )

    def test_rank_diagnostics_detect_low_rank_ld(self):
        eigenvalues = np.array([3.0, 1.0, 1e-12])
        block = np.diag(eigenvalues)
        diagnostics = diagnose_ld_blocks(
            [block], [3], bucket_size=1, rank_rtol=1e-8
        )
        self.assertEqual(diagnostics["rank_min"], 2)
        self.assertEqual(diagnostics["rank_max"], 2)
        self.assertAlmostEqual(diagnostics["rank_fraction_median"], 2/3)
        self.assertIn("numerical rank", format_ld_diagnostics(diagnostics))

    def test_eigen_projection_matches_legacy_svd_projection(self):
        matrix = np.array([
            [1.0, 1.2, 0.1],
            [1.2, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        _, singular_values, right = linalg.svd(matrix)
        legacy = (
            matrix + right.T @ np.diag(singular_values) @ right
        ) / 2.0
        projected, factor, eigenvalues = _project_ld_psd(matrix)
        np.testing.assert_allclose(projected, legacy, rtol=1e-12,
                                   atol=1e-12)
        np.testing.assert_allclose(factor @ factor.T, projected,
                                   rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(float(eigenvalues.min()), 0.0)

    def test_ld_parser_reuses_psd_factor_for_pcg(self):
        with tempfile.TemporaryDirectory(prefix="ldblk_1kg_") as directory:
            filename = os.path.join(directory, "ldblk_1kg_chr22.hdf5")
            with h5py.File(filename, "w") as handle:
                group = handle.create_group("blk_1")
                group.create_dataset(
                    "ldblk",
                    data=np.array([
                        [1.0, 0.2, 0.1],
                        [0.2, 1.0, 0.3],
                        [0.1, 0.3, 1.0],
                    ]),
                )
                group.create_dataset(
                    "snplist", data=np.asarray([b"rs1", b"rs2", b"rs3"])
                )

            sst = {"SNP": ["rs1", "rs3"], "FLP": [1, -1]}
            blocks, sizes, factors, eigenvalues = parse_ldblk(
                directory, sst, 22, return_factors=True
            )

        self.assertEqual(sizes, [2])
        np.testing.assert_allclose(
            factors[0] @ factors[0].T, blocks[0],
            rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.sort(np.sum(factors[0] ** 2, axis=0)),
            np.sort(eigenvalues[0]),
            rtol=1e-12, atol=1e-12,
        )


@unittest.skipUnless(_cuda_available(), "CuPy and a CUDA device are required")
class CudaBetaBackendTests(unittest.TestCase):
    def test_direct_backend_reports_cholesky_failure(self):
        backend = CudaDirectBetaBackend(
            [-2.0 * np.eye(2)],
            [2],
            np.ones((2, 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )
        with self.assertRaisesRegex(
                RuntimeError, "direct CUDA Cholesky failed"):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_direct_backend_matches_generic_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        generic = CudaBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        direct = CudaDirectBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )

        expected_beta, expected_quad = generic.sample(psi, sigma)
        actual_beta, actual_quad = direct.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-11, atol=1e-11
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn("potrfBatched", direct.describe())

    def test_hybrid_backend_matches_direct_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        direct = CudaDirectBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        hybrid = CudaHybridBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )

        expected_beta, expected_quad = direct.sample(psi, sigma)
        actual_beta, actual_quad = hybrid.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-10, atol=1e-10
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn("regular potrf/trsm for 3 matrices", hybrid.describe())

    def test_hybrid_backend_keeps_dense_buckets_batched(self):
        blocks = [np.eye(2)] * 8
        backend = CudaHybridBetaBackend(
            blocks,
            [2] * len(blocks),
            np.ones((2 * len(blocks), 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )

        self.assertIn("batched for 8 matrices in 1 buckets", backend.describe())

    def test_hybrid_backend_reports_cholesky_failure(self):
        backend = CudaHybridBetaBackend(
            [-2.0 * np.eye(2)],
            [2],
            np.ones((2, 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
        )
        with self.assertRaisesRegex(
                RuntimeError, "hybrid CUDA Cholesky failed"):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_stream_backend_matches_hybrid_cuda_draw(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        hybrid = CudaHybridBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        streamed = CudaStreamsBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
            cuda_streams=2,
        )

        expected_beta, expected_quad = hybrid.sample(psi, sigma)
        actual_beta, actual_quad = streamed.sample(psi, sigma)

        np.testing.assert_allclose(
            actual_beta, expected_beta, rtol=1e-10, atol=1e-10
        )
        self.assertAlmostEqual(actual_quad, expected_quad, places=10)
        self.assertIn("2 concurrent streams", streamed.describe())

    def test_stream_backend_is_seeded_reproducibly(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        backends = [
            CudaStreamsBetaBackend(
                blocks,
                sizes,
                beta_mrg,
                1000,
                seed=987,
                cuda_bucket_size=4,
                cuda_streams=2,
            )
            for _ in range(2)
        ]

        first_beta, first_quad = backends[0].sample(psi, 0.7)
        second_beta, second_quad = backends[1].sample(psi, 0.7)

        np.testing.assert_array_equal(first_beta, second_beta)
        self.assertEqual(first_quad, second_quad)

    def test_stream_backend_reports_cholesky_failure(self):
        backend = CudaStreamsBetaBackend(
            [-2.0 * np.eye(2)],
            [2],
            np.ones((2, 1)),
            1000,
            seed=123,
            cuda_bucket_size=1,
            cuda_streams=2,
        )
        with self.assertRaisesRegex(
                RuntimeError, "streamed CUDA Cholesky failed"):
            backend.sample(np.ones((2, 1)), 1.0)

    def test_fp32_backend_tracks_analytic_conditional_mean(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        backend = CudaFp32BetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
        )
        beta, quad = backend.sample(psi, 0.0)

        expected_beta = np.empty_like(beta_mrg)
        expected_quad = 0.0
        start = 0
        for ld, size in zip(blocks, sizes):
            if not size:
                continue
            block_slice = slice(start, start + size)
            precision = ld + np.diag(1.0 / psi[block_slice, 0])
            expected_beta[block_slice] = np.linalg.solve(
                precision, beta_mrg[block_slice]
            )
            expected_quad += (
                expected_beta[block_slice].T @ precision @
                expected_beta[block_slice]
            ).item()
            start += size

        np.testing.assert_allclose(
            beta, expected_beta, rtol=2e-5, atol=2e-6
        )
        self.assertAlmostEqual(
            quad / expected_quad, 1.0, places=5
        )
        self.assertIn("approximate", backend.describe())

    def test_fp32_backend_matches_conditional_moments(self):
        ld = np.array([[1.0, 0.25], [0.25, 0.8]])
        beta_mrg = np.array([[0.1], [-0.2]])
        psi = np.array([[0.4], [0.9]])
        sigma = 0.7
        n_gwas = 100
        precision = ld + np.diag(1.0 / psi[:, 0])
        expected_mean = np.linalg.solve(precision, beta_mrg)[:, 0]
        expected_scale = sigma / n_gwas
        chol = np.linalg.cholesky(precision)

        backend = CudaFp32BetaBackend(
            [ld], [2], beta_mrg, n_gwas,
            seed=321, cuda_bucket_size=1,
        )
        draws = np.empty((2048, 2))
        for index in range(draws.shape[0]):
            draws[index] = backend.sample(psi, sigma)[0][:, 0]

        standardized = (
            (draws - expected_mean) @ chol / np.sqrt(expected_scale)
        )
        self.assertLess(float(np.max(np.abs(standardized.mean(axis=0)))),
                        0.08)
        covariance = np.cov(standardized, rowvar=False, bias=True)
        self.assertLess(float(np.max(np.abs(covariance - np.eye(2)))),
                        0.10)

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

    def test_pcg_irregular_blocks_converge_to_true_quadratic_form(self):
        blocks, sizes, beta_mrg, psi = _inputs()
        sigma = 0.7
        backend = CudaPcgBetaBackend(
            blocks,
            sizes,
            beta_mrg,
            1000,
            seed=123,
            cuda_bucket_size=4,
            pcg_tol=1e-11,
            pcg_maxiter=100,
            pcg_check_interval=2,
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
        self.assertIn("maximum true relative residual", backend.profile_summary())


if __name__ == "__main__":
    unittest.main()
