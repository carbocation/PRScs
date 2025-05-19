#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""

import gigrnd

import numpy as np
from scipy import linalg
from scipy.stats import geninvgauss
from joblib import Parallel, delayed
from joblib import parallel
from threadpoolctl import threadpool_limits
import torch
from torch import Tensor
import time, collections

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,                  # change to DEBUG for finer detail
    format='%(asctime)s  %(levelname)s  %(message)s',
    stream=sys.stdout,                   # ensures Docker/cluster stdout sees it
    force=True                           # overrides any prior config
)
log = logging.getLogger(__name__)

# ---------- helper for one LD block ----------
def _sample_block_torch(
    ld_block      : Tensor,          # (m,m)  LD matrix for the block
    psi_block     : Tensor,          # (m,1)  local shrinkage
    beta_mrg_block: Tensor,          # (m,1)  marginal β̂
    sigma         : float,
    n_gwas        : int,
    *, generator: torch.Generator
) -> tuple[Tensor,float]:
    """
    Single-block β draw on *any* device (CPU or CUDA).

    Returns (beta_block, quadratic_form) where

        quadratic_form = βᵀ (LD + diag(1/ψ)) β
    """
    # 1. K⁻¹ = LD + diag(1/ψ)
    dinvt = ld_block + torch.diag_embed(1.0 / psi_block.squeeze(-1))

    # 2. Cholesky
    chol_u = torch.linalg.cholesky(dinvt)         # upper-triangular (U)

    # 3. RHS term  Uᵀ⁻¹ β̂           (solve with lower-triangular system)
    rhs = torch.linalg.solve_triangular(
        chol_u.mT,                                # Uᵀ is lower
        beta_mrg_block,
        upper=False
    )

    # 4. Add normal noise 𝒩(0, σ/n I)
    z = torch.randn_like(rhs, generator=generator) * (sigma / n_gwas) ** 0.5

    # 5. Final solve  β = U⁻¹ (rhs + z)
    beta_block = torch.linalg.solve_triangular(
        chol_u,
        rhs + z,
        upper=True
    )

    quad = (beta_block.T @ dinvt @ beta_block).item()
    return beta_block, quad

def mcmc(
    a, b, phi_init,
    sst_dict, n_gwas,
    ld_blocks, block_sizes,
    n_iter, n_burnin, thin,
    chrom, out_dir,
    beta_std='FALSE', write_psi='FALSE', write_pst='FALSE',
    seed: int | None = None,
    *, device: str | torch.device = 'cpu'
):
    """
    PyTorch replacement for `mcmc()`.

    * All heavy linear-algebra happens on the chosen device.
    * ψ-updates still call `gigrnd.gig_rvs_vec` on CPU in-place.
    """

    # -----------------------------------------------------------------------
    # 0.  PREP
    # -----------------------------------------------------------------------
    torch.set_default_dtype(torch.float64)
    dev   = torch.device(device)
    rng   = torch.Generator(device='cpu').manual_seed(seed or 0)   # CPU for ψ
    rng_t = torch.Generator(device=dev).manual_seed(seed or 0)     # β-block RNG

    beta_mrg = torch.as_tensor(sst_dict['BETA'],  device=dev).reshape(-1,1)
    maf      = torch.as_tensor(sst_dict['MAF'],   device='cpu').reshape(-1,1)
    p        = beta_mrg.numel()
    n_pst    = (n_iter - n_burnin)//thin

    # block indexing (same as before)
    starts     = [0] + list(torch.tensor(block_sizes).cumsum(0)[:-1])
    idx_ranges = [slice(s, s+sz) for s,sz in zip(starts, block_sizes)]

    # -----------------------------------------------------------------------
    # 1.  STATE  (all torch tensors unless noted)
    # -----------------------------------------------------------------------
    beta       = torch.zeros((p,1), device=dev)
    psi        = torch.ones ((p,1), device='cpu')     # stay on CPU for GIG
    sigma      = torch.tensor(1.0, device='cpu')

    if phi_init is None:
        phi, update_phi = torch.tensor(1.0), True
    else:
        phi, update_phi = torch.tensor(float(phi_init)), False

    # running means
    beta_est  = torch.zeros_like(beta)
    psi_est   = torch.zeros_like(psi)
    sigma_est = torch.tensor(0.0)
    phi_est   = torch.tensor(0.0)

    if write_pst.upper() == 'TRUE':
        beta_pst = torch.empty((p, n_pst), device='cpu')
        pst_idx  = 0

    # -----------------------------------------------------------------------
    # 2.  MCMC LOOP
    # -----------------------------------------------------------------------
    for itr in range(1, n_iter+1):
        # --- 2.1  β-block updates (device = dev) ---------------------------
        quad_total = 0.0
        for k, slc in enumerate(idx_ranges):
            if block_sizes[k] == 0:
                continue

            beta_b, quad_b = _sample_block_torch(
                ld_blocks[k].to(dev),
                psi[slc].to(dev),
                beta_mrg[slc],
                float(sigma), int(n_gwas),
                generator=rng_t
            )
            beta[slc] = beta_b
            quad_total += quad_b
        # -------------------------------------------------------------------

        # --- 2.2  global σ update (CPU) ------------------------------------
        #   e1 = n/2 (1 − 2β·β̂ + βᵀLDβ)
        #   e2 = n/2 Σ β²/ψ
        s1  = (beta.to('cpu') * beta_mrg.to('cpu')).sum().item()
        s2  = ((beta.to('cpu')**2) / psi).sum().item()
        e1  = 0.5*n_gwas*(1.0 - 2.0*s1 + quad_total)
        e2  = 0.5*n_gwas*s2
        err = max(e1, e2)

        sigma = 1.0 / np.random.gamma((n_gwas + p)*0.5, 1.0/err)   # numpy OK

        # --- 2.3  δ & ψ updates  (CPU) -------------------------------------
        delta = np.random.gamma(a+b, 1.0/(psi.numpy() + phi))
        gigrnd.gig_rvs_vec(                    # in-place on psi.numpy()
            psi.numpy().ravel(),
            a - 0.5,
            delta.ravel(),
            beta.to('cpu').numpy().ravel(),
            sigma,
            n_gwas
        )
        psi.clamp_(max=1.0)

        # --- 2.4  φ update --------------------------------------------------
        if update_phi:
            w   = np.random.gamma(1.0, 1.0/(phi + 1.0))
            phi = torch.tensor(np.random.gamma(p*b + 0.5, 1.0/(delta.sum() + w)))

        # --- 2.5  accumulate posterior samples -----------------------------
        if itr > n_burnin and itr % thin == 0:
            weight = 1.0 / n_pst
            beta_est  += beta * weight
            psi_est   += psi  * weight
            sigma_est += sigma* weight
            phi_est   += phi  * weight
            if write_pst.upper() == 'TRUE':
                beta_pst[:, pst_idx] = beta.to('cpu')
                pst_idx += 1

        # --- 2.6  logging ---------------------------------------------------
        status = "burn-in" if itr < n_burnin else f"φ={phi_est.item():.3e}"
        log.info("chr %d  iteration %d/%d  %s", chrom, itr, n_iter, status)

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if update_phi == True:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

    with open(eff_file, 'w') as ff:
        if write_pst == 'TRUE':
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_pst):
                ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta))
        else:
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if update_phi == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi))

    # print estimated phi
    if update_phi == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')


