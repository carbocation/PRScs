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

def make_toy_inputs(p: int = 500, n_blocks: int = 10, rho: float = 0.9):
    """
    Create a toy dataset split into `n_blocks` contiguous LD blocks.
    Each block gets its own correlation matrix; between-block LD is zero
    (exactly what PRS-CS expects).

    Returns
    -------
    sst_dict   : GWAS summary stats (same as before)
    ld_blocks  : list[np.ndarray]   – one LD matrix per block
    block_sizes: list[int]          – sizes of the LD blocks
    """
    assert 1 <= n_blocks <= p, "`n_blocks` must be between 1 and p"

    # ― summary statistics ---------------------------------------------------
    rng = np.random.default_rng(42)

    sst_dict = {
        "SNP":  [f"rs{idx}" for idx in range(p)],
        "BP":   list(range(1, p + 1)),
        "A1":   ["A"] * p,
        "A2":   ["C"] * p,
        "BETA": rng.standard_normal(p) * 0.01,
        "MAF":  rng.uniform(0.05, 0.5, p),
    }

    # ― split p SNPs as evenly as possible -----------------------------------
    base   = p // n_blocks
    sizes  = [base] * n_blocks
    sizes[-1] += p - base * n_blocks          # put any remainder in the last

    ld_blocks: list[np.ndarray] = []
    for m in sizes:
        coords = np.arange(m)
        dist   = np.abs(coords[:, None] - coords[None, :])
        ld_blocks.append((rho ** dist).astype(np.float64))

    return sst_dict, ld_blocks, sizes

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
        n_iter      = 2000,
        n_burnin    = 200,
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
