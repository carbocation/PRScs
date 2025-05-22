#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""

import gigrnd

import numpy as np
from scipy import linalg
from joblib import Parallel, delayed
from joblib import parallel
from threadpoolctl import threadpool_limits

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
def _sample_block_wrapped(state, psi_blk, beta_mrg_blk, sigma, n, block_seed=None):
    """Restrict MKL threads inside each joblib worker."""
    with threadpool_limits(limits=2, user_api="blas"):
        return _sample_block(state, psi_blk, beta_mrg_blk, sigma, n, block_seed)

def _sample_block(state, psi_blk, beta_mrg_blk, sigma, n, block_seed=None):
    """
    Draw β for one LD block and return (β_block, quadratic form).
    """
    rng = np.random.default_rng(block_seed)

    # --- fast diagonal update ---------------------------------------------
    diag_new = (1.0 / psi_blk).astype(state['L'].dtype, copy=False)
    gigrnd.chol_diag_update_inplace(state['L'], state['diag_curr'], diag_new)
    # state['diag_curr'] was updated in-place by the kernel

    # --- Gibbs draw --------------------------------------------------------
    z = rng.standard_normal((len(psi_blk), 1)) * np.sqrt(sigma / n)
    beta_b = linalg.solve_triangular(
        state['L'],
        linalg.solve_triangular(state['L'], beta_mrg_blk, trans='T') + z,
        trans='N'
    )
    # more expensive quadratic form:
    # quad_b = float(beta_b.T @ (state['L'] @ state['L'].T) @ beta_b)
    # cheaper quadratic form: || Lᵀ β ||²
    tmp = state['L'].T @ beta_b  # This computes Lᵀβ
    quad_b = float((tmp * tmp).sum())
    return beta_b, quad_b

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')

    n_jobs = int(os.environ.get("PRSCS_N_JOBS", "1"))     # default: 1
    UNCLAMP_PSI = os.getenv("PRSCS_UNCLAMP_PSI", "false").lower() in ("1", "true", "yes", "on")

    # seed
    if seed is not None:
        np.random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    if n_pst == 0:
        raise ValueError(
            f"n_iter={n_iter}, n_burnin={n_burnin}, thin={thin} ⇒ 0 posterior samples; "
            "choose larger n_iter or smaller thin."
        )
    p = len(sst_dict['SNP'])

    # --- block index bookkeeping ---
    starts = np.cumsum([0] + blk_size[:-1])       # 0-based starts
    idx_ranges = [range(s, s + sz) for s, sz in zip(starts, blk_size)]

    # --- persistent Cholesky state for every LD block -----------------
    block_state = []
    for k, r in enumerate(idx_ranges):
        if blk_size[k] == 0:                # empty block
            block_state.append(None)
            continue
        L0 = linalg.cholesky(ld_blk[k] + np.eye(blk_size[k]), lower=True)
        d0 = np.ones(blk_size[k], dtype=L0.dtype)           # current 1/ψ (starts at 1)
        block_state.append({'L': L0, 'diag_curr': d0})

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0

    beta_1d = beta[:, 0]    # view – updates automatically
    psi_1d  = psi[:, 0]     # view – output buffer
    
    if phi is None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    # Note that with our approach to state management (specifically "L"), the
    # block state *will not evolve* (will not propagate back from the workers)
    # if you use a truly parallel / 'loky' backend.
    parpool = Parallel(n_jobs=n_jobs, backend="threading")
    
    # MCMC
    pp = 0
    timer  = collections.Counter()
    counts = collections.Counter()
    for itr in range(1,n_iter+1):
        loop_start = time.perf_counter()
        # --- parallel block sampler -------------------
        t0 = time.perf_counter()
        active = [(k, r) for k, r in enumerate(idx_ranges) if blk_size[k] > 0]
        results = parpool(
                    delayed(_sample_block_wrapped)(block_state[k],
                                        psi[r, 0],
                                        beta_mrg[r],
                                        sigma, 
                                        n,
                                        block_seed=(None if seed is None else seed + itr * 1_000_003 + k))
                    for k, r in active
                )
        timer['beta'] += time.perf_counter() - t0
        counts['beta'] += 1
        
        if itr == 1:  # only on first iteration
            log.info(f"[DBG] backend: {parpool._backend.__class__.__name__}, "
                f"n_jobs={n_jobs}, non-empty blocks={len(active)}")

        quad = 0.0
        for (r, (beta_b, quad_b)) in zip([r for _, r in active], results):
            beta[r] = beta_b
            quad   += quad_b
        # ----------------------------------------------------

        s1 = float((beta * beta_mrg).sum())
        s2 = float((beta**2 / psi).sum())
        e1 = float(n/2.0*(1.0 - 2.0*s1 + quad))
        e2 = float(n/2.0*s2)
        err = max(e1, e2)

        # force sigma to be a Python float (not a 0-d array)
        sigma = float(1.0/np.random.gamma((n+p)/2.0, 1.0/err))

        delta = np.random.gamma(a+b, 1.0/(psi+phi))

        # ---------- ψ-update ----------
        t0 = time.perf_counter()
        gigrnd.gig_rvs_vec(
                psi_1d,          # out
                a - 0.5,
                delta[:, 0],
                beta_1d,
                sigma,
                n
        )
        if not UNCLAMP_PSI:
            psi_1d[psi_1d > 1.0] = 1.0
        timer['psi'] += time.perf_counter() - t0
        counts['psi'] += 1
        # ------------------------------

        if phi_updt == True:
            w = float(np.random.gamma(1.0, 1.0/(phi+1.0)))
            phi = float(np.random.gamma(p*b+0.5, 1.0/(sum(delta)+w)))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1
        
        timer['loop'] += time.perf_counter() - loop_start
        counts['loop'] += 1        # same as number of iterations
        
        b  = timer['beta'] / max(counts['beta'], 1)
        ps = timer['psi']  / max(counts['psi'],  1)
        lo = timer['loop'] / max(counts['loop'], 1)
        if itr < n_burnin:
            status = "burning in"
        else:
            status = (
                f"σ resid={sigma:.3e}, "
                f"φ=(inst {phi:.3e} | post {phi_est:.3e}), "
                f"snp ψ̄={float(psi.mean()):.3e}"
            )
        log.info(f"[PROFILE chr{chrom}] iter {itr:4d} | "
            f"β {b:6.3f}s  ψ {ps:6.3f}s  other {lo-b-ps:6.3f}s (tot {lo:6.3f}s) | {status}")

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if phi_updt == True:
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
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')


