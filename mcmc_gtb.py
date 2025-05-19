#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""

import gigrnd

import numpy as np
from scipy import linalg
from scipy.stats import geninvgauss
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor

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
def _sample_block(ld, psi_blk, beta_mrg_blk, sigma, n, block_seed=None):
    """Draw β for one LD block and return (β_block, quadratic form)."""
    rng = np.random.default_rng(block_seed)
    dinvt = ld + np.diag(1.0 / psi_blk)
    chol  = linalg.cholesky(dinvt)
    z     = rng.standard_normal((len(psi_blk), 1)) * np.sqrt(sigma / n)
    beta_b = linalg.solve_triangular(
        chol,
        linalg.solve_triangular(chol, beta_mrg_blk, trans='T') + z,
        trans='N'
    )
    quad_b = float(beta_b.T @ dinvt @ beta_b)
    return beta_b, quad_b

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')

    n_jobs = int(os.environ.get("PRSCS_N_JOBS", "1"))     # default: 1
    PSI_CHUNK = 10_000                                        # ~20 k SNPs per thread-call

    # seed
    if seed is not None:
        np.random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    p = len(sst_dict['SNP'])
    n_blk = len(ld_blk)

    # --- block index bookkeeping ---
    starts = np.cumsum([0] + blk_size[:-1])       # 0-based starts
    idx_ranges = [range(s, s + sz) for s, sz in zip(starts, blk_size)]

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0
    
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    with threadpool_limits(limits=1, user_api="blas"):   # lock BLAS to 1 thread for creating the parpool
        parpool = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")
    
    thread_pool_psi = ThreadPoolExecutor(max_workers=n_jobs)

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        # --- parallel block sampler -------------------
        active = [(k, r) for k, r in enumerate(idx_ranges) if blk_size[k] > 0]
        results = parpool(
                    delayed(_sample_block)(ld_blk[k],
                                        psi[r, 0],
                                        beta_mrg[r],
                                        sigma, 
                                        n,
                                        block_seed=(None if seed is None else seed + itr * 1_000_003 + k))
                    for k, r in active
                )

        if itr % 1 == 0:
            log.info('chr %d  started iteration %d of %d', chrom, itr, n_iter)
            print(f"[DEBUG] chr {chrom} completed iteration {itr} of {n_iter} with n_jobs={n_jobs} and non-empty blocks={len(active)}")

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

        # ---------- threaded ψ-update ----------
        # one deterministic RNG for this iteration (keeps reproducible across n_jobs)
        rng_iter = None if seed is None else np.random.default_rng(seed + itr * 2_000_033)

        def draw_chunk(start, stop, rs):
            """Draw ψ for slice [start:stop)."""
            return geninvgauss.rvs(
                a - 0.5,
                2.0 * delta[start:stop, 0],
                scale = sigma / (n * (delta[start:stop, 0] ** 2)),
                size  = stop - start,
                random_state = rs
            )

        # build slice list
        idxs = list(range(0, p, PSI_CHUNK)) + [p]       # e.g. 0, 20k, 40k, …, p
        slices = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]

        # run chunks in a thread pool (SciPy releases the GIL, so threads scale)
        futs = [ thread_pool_psi.submit(draw_chunk, s, e,
                None if rng_iter is None
                    else np.random.default_rng(int(rng_iter.integers(1<<63))))
                for s, e in slices ]
        psi[:, 0] = np.concatenate([f.result() for f in futs])

        psi[psi > 1.0] = 1.0
        # ---------------------------------------------------------------

        if phi_updt == True:
            w = np.random.gamma(1.0, 1.0/(phi+1.0))
            phi = np.random.gamma(p*b+0.5, 1.0/(sum(delta)+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

    thread_pool_psi.shutdown(wait=True)

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


