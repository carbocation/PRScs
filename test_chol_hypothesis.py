# test_chol_hypothesis.py  –  “extreme torture” version
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis.extra.numpy import arrays

from mcmc_gtb import _sample_block_fused

# ─ helpers ───────────────────────────────────────────────────────────────
def _make_pathological_ld(m, *, seed=0):
    """Return an SPD matrix with condition number ≈ 1e20."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    eigs = np.logspace(-20, 0, m)          # 1e-20 … 1
    return Q @ np.diag(eigs) @ Q.T

def _fresh_state(ld_sub, *, seed=0):
    m = ld_sub.shape[0]
    return dict(
        ld_blk    = ld_sub,
        beta_mrg  = np.zeros(m),
        rng       = np.random.default_rng(seed),
        L         = np.linalg.cholesky(ld_sub + np.eye(m)).copy(order="F"),
        diag_curr = np.ones(m),
        invpsi    = np.empty(m),
        zbuf      = np.empty(m),
        work      = np.empty(m),
    )

# ─ property test ─────────────────────────────────────────────────────────
@given(
    m     = st.integers(4, 64),                # bigger blocks
    k     = st.integers(1, 20),                # successive updates per block
    data  = st.data(),
)
@settings(
    deadline=None,
    max_examples=2000,
    suppress_health_check=[
        HealthCheck.too_slow, HealthCheck.filter_too_much
    ],
)
def test_extreme_chol_updates_survive(m, k, data):
    """No NaN/Inf/exception under worst-case incremental updates."""
    state = _fresh_state(_make_pathological_ld(m))
    sigma, n = 1.0, 1_000_000_000            # gigantic n ⇒ tiny σ/n

    psi_strategy = arrays(
        np.float64, (m,),
        elements=st.floats(-800, 800, allow_nan=False, allow_infinity=False)
    )

    for _ in range(k):
        # draw a fresh ψ, exponentiate → log-uniform in [1e-800, 1e800]
        psi_slice = np.exp(data.draw(psi_strategy))
        beta_blk, qform = _sample_block_fused(state, psi_slice, sigma, n)

        assert np.isfinite(beta_blk).all(),  "non-finite β returned"
        assert np.isfinite(qform),           "non-finite quadratic form"
