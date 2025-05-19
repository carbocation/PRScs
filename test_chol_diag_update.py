# test_chol_diag_update.py
"""
Stress-test for the incremental Cholesky updater used by _sample_block_fused.

The idea is to feed it:
  • extremely ill-conditioned LD sub-matrices
  • ψ-vectors that swing wildly between PSI_MIN and PSI_MAX
so that every rank-1 update, downdate, and early-bailout branch
is exercised in a few seconds.

Run with:   pytest -q test_chol_diag_update.py
"""

import numpy as np
import pytest

from mcmc_gtb import _sample_block_fused, PSI_MIN, PSI_MAX

# -------------------------------------------------------------------------
def _make_pathological_ld(m: int, *, seed: int = 0) -> np.ndarray:
    """
    Return an m×m SPD matrix with eigenvalues spanning 10¹² – i.e. about as
    ill-conditioned as double precision can handle before outright singularity.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))      # random orthogonal
    eigs = np.logspace(-12, 0, m)                         # 1e-12 … 1
    return Q @ np.diag(eigs) @ Q.T


def _fresh_state(ld_sub: np.ndarray, *, seed: int = 0) -> dict:
    """Build the minimal block_state dict expected by _sample_block_fused."""
    m = ld_sub.shape[0]
    L0 = np.linalg.cholesky(ld_sub + np.eye(m)).copy(order="F")

    return dict(
        ld_blk     = ld_sub,
        beta_mrg   = np.zeros(m),          # zero mean ⇒ we only test the update path
        rng        = np.random.default_rng(seed),
        L          = L0,
        diag_curr  = np.ones(m),           # invψ = 1 for every SNP initially
        invpsi     = np.empty(m),
        zbuf       = np.empty(m),
        work       = np.empty(m),
    )


@pytest.mark.parametrize("m", [4, 8, 16])
def test_incremental_update_survives_pathological_inputs(m):
    """
    Run 1 000 incremental updates on a single ill-conditioned LD block,
    drawing a brand-new ψ-vector each time from a log-uniform distribution
    that covers *far* beyond the production clip interval.

    Expect: no exceptions, all β’s finite.
    """
    state = _fresh_state(_make_pathological_ld(m), seed=123)
    sigma, n = 1.0, 10_000

    rng = np.random.default_rng(321)
    for _ in range(1_000):
        # log-uniform on [1e-12, 1e+12] – intentionally outside PSI_[MIN,MAX]
        psi_slice = np.exp(rng.uniform(-12, 12, size=m))
        # With 5 % probability force a huge downdate on the first pivot
        if rng.random() < 0.05:
            psi_slice[0] = PSI_MAX * 1e4

        beta_block, qform = _sample_block_fused(state, psi_slice, sigma, n)
        assert np.isfinite(beta_block).all()
        assert np.isfinite(qform)
