# test_prscs_smoke.py
"""
Run a 10-iteration PRS-CS loop on a 200-SNP toy dataset.
Purpose: catch import/typing/NaN explosions as part of CI.
"""

import os, tempfile, shutil
import numpy as np
from pathlib import Path

# --- import the code under test -------------------------------------------
from mcmc_gtb import mcmc


def make_toy_inputs(p: int = 200):
    """Return (sst_dict, ld_blocks, block_sizes) for a single-block toy run."""
    rng = np.random.default_rng(42)

    sst_dict = {
        "SNP":  [f"rs{idx}"    for idx in range(p)],
        "BP":   [idx + 1       for idx in range(p)],
        "A1":   ["A"] * p,
        "A2":   ["C"] * p,
        "BETA": rng.standard_normal(p) * 0.01,          # tiny marginal effects
        "MAF":  rng.uniform(0.05, 0.5, p),
    }

    # simple LD matrix: exponential decay with distance
    coords = np.arange(p)
    dist   = np.abs(coords[:, None] - coords[None, :])
    ld_mat = 0.9 ** dist                                   # ρ^|i-j|

    ld_blocks  = [ld_mat.astype(np.float64)]
    block_sizes = [p]

    return sst_dict, ld_blocks, block_sizes


def test_prscs_smoke():
    sst, ld_blk, blk_size = make_toy_inputs()

    # work in a temp dir so files don’t clutter the repo
    tmpdir = Path(tempfile.mkdtemp())
    out_prefix = tmpdir / "toy"

    # run a *very* short chain — just enough to shake out shape issues
    mcmc(
        a           = 1.0,
        b           = 0.5,
        phi         = None,         # let the sampler learn φ
        sst_dict    = sst,
        n           = 1_000,        # synthetic sample size
        ld_blk      = ld_blk,
        blk_size    = blk_size,
        n_iter      = 10,
        n_burnin    = 2,
        thin        = 1,
        chrom       = 22,
        out_dir     = str(out_prefix),
        beta_std    = "FALSE",
        write_psi   = "TRUE",
        write_pst   = "FALSE",
        seed        = 123
    )

    # ---- assertions -------------------------------------------------------
    # load posterior means that mcmc() just wrote
    beta_file = next(tmpdir.glob("*_pst_eff_*_chr22.txt"))
    betas = np.loadtxt(beta_file, usecols=-1)       # last column is β̂
    assert np.isfinite(betas).all(),  "NaN/Inf in posterior betas"

    psi_file = next(tmpdir.glob("*_pst_psi_*_chr22.txt"))
    psis = np.loadtxt(psi_file, usecols=1)
    assert (psis > 0).all(),        "non-positive ψ encountered"

    # clean up
    shutil.rmtree(tmpdir)
