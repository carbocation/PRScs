#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""

import gigrnd

import numpy as np
from scipy import linalg
from scipy.linalg import cholesky_update
from joblib import Parallel, delayed
import joblib
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
def _solve_with_cached_chol(chol, dinvt, beta_mrg_blk, sigma, n, rng):
    """chol and dinvt are float32; beta_mrg_blk is float64."""

    with threadpool_limits(limits=1,user_api="blas"):
        beta_m32 = beta_mrg_blk.astype(np.float32, copy=False)

        z   = rng.standard_normal(beta_m32.shape).astype(np.float32) * np.sqrt(sigma / n)
        tmp = linalg.solve_triangular(chol, beta_m32, lower=True, check_finite=False)
        beta_b_f32 = linalg.solve_triangular(chol.T, tmp + z, lower=False, check_finite=False)

        quad_b = float(beta_b_f32.astype(np.float64).T @
                    dinvt.astype(np.float64) @
                    beta_b_f32.astype(np.float64))
        return beta_b_f32, quad_b

# ---- helper ----------------------------------------------------
def _diag_chol_update(chol, dinvt, invdiag, delta):
    """
    Apply the change `delta` to the *diagonal* of D⁻¹ and update
    the cached Cholesky factor `chol` in-place.

    Parameters
    ----------
    chol   : ndarray (m, m)  – lower-triangular Cholesky factor   (float32)
    dinvt  : ndarray (m, m)  – cached D⁻¹                         (float32)
    invdiag: ndarray (m,)    – cached diag(D⁻¹)                   (float32)
    delta  : ndarray (m,)    – new − old diagonal                 (float32/64)
    """
    if not np.any(delta):
        return                         # nothing to do

    # split signs because `sign` is scalar in scipy.linalg.cholesky_update
    for sign_val, mask in ((+1, delta > 0), (-1, delta < 0)):
        if not np.any(mask):
            continue
        cols = mask.sum()
        # U has shape (m,  r), r = #diagonal elements of this sign
        U = np.zeros((chol.shape[0], cols), dtype=chol.dtype, order='F')
        idx = np.nonzero(mask)[0]
        U[idx, np.arange(cols)] = np.sqrt(np.abs(delta[mask]))
        cholesky_update(
            chol,
            U,
            lower=True,
            overwrite_c=True,
            check_finite=False,
            sign=sign_val,
        )

    # keep the caches consistent
    dinvt[np.diag_indices_from(dinvt)] += delta.astype(dinvt.dtype)
    invdiag[:] += delta.astype(invdiag.dtype)

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')

    n_jobs = int(os.environ.get("PRSCS_N_JOBS", "1"))     # default: 1

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

    chol_blk, invdiag_blk, dinvt_blk = [], [], []
    for k, r in enumerate(idx_ranges):
        dinvt = ld_blk[k].astype(np.float32, copy=True)
        dinvt[np.diag_indices_from(dinvt)] += 1.0
        chol_blk.append(linalg.cholesky(dinvt, lower=True, check_finite=False))
        invdiag_blk.append(np.ones(len(r), dtype=np.float32))
        dinvt_blk.append(dinvt)

    parpool = Parallel(n_jobs=n_jobs, backend="threading", require="sharedmem")

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0
    
    # MCMC
    pp = 0
    timer  = collections.Counter()
    counts = collections.Counter()
    for itr in range(1,n_iter+1):
        loop_start = time.perf_counter()
        # --- parallel block sampler -------------------
        t0 = time.perf_counter()
        # 1) pick active blocks
        active = [(k, r) for k, r in enumerate(idx_ranges) if blk_size[k] > 0]

        # 2) ------- rank-1 updates of each Cholesky -----------------
        for k, r in active:
            delta_vec = (1.0 / psi[r, 0]) - invdiag_blk[k]    # shape (m,)
            _diag_chol_update(
                chol_blk[k],
                dinvt_blk[k],
                invdiag_blk[k],
                delta_vec,
            )

        # 3) ------- draw β in parallel using cached factors --------------
        t0 = time.perf_counter()
        results = parpool(
            delayed(_solve_with_cached_chol)(
                chol_blk[k],
                dinvt_blk[k],
                beta_mrg[r],
                sigma,
                n,
                np.random.default_rng(None if seed is None else (seed + itr*1_000_003 + k) & 0xFFFFFFFFFFFFFFFF)
            )
            for k, r in active
        )
        # copy results back
        quad = 0.0
        for (r, (beta_b_f32, quad_b)) in zip([r for _, r in active], results):
            beta[r] = beta_b_f32.astype(np.float64)   # master β stays FP64
            quad   += quad_b
        
        timer['beta'] += time.perf_counter() - t0
        counts['beta'] += 1
        
        if itr == 1:  # only on first iteration
            backend = joblib.parallel.get_active_backend()[0]
            print(f"[DBG] backend: {backend.__class__.__name__}, "
                f"n_jobs={n_jobs}, non-empty blocks={len(active)}")

        if itr % 1 == 0:
            status_of_phi = "burning in" if itr < n_burnin else f"φ={float(phi_est):.3e}"
            log.info('chr %d  started iteration %d of %d (%s)', chrom, itr, n_iter, status_of_phi)
            print(f"[DEBUG] chr {chrom} completed iteration {itr} of {n_iter} with n_jobs={n_jobs} and non-empty blocks={len(active)}, {status_of_phi}")

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
        psi_1d[psi_1d > 1.0] = 1.0
        timer['psi'] += time.perf_counter() - t0
        counts['psi'] += 1
        # ------------------------------

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
        
        timer['loop'] += time.perf_counter() - loop_start
        counts['loop'] += 1        # same as number of iterations
        
        timer_b  = timer['beta'] / max(counts['beta'], 1)
        timer_ps = timer['psi']  / max(counts['psi'],  1)
        timer_lo = timer['loop'] / max(counts['loop'], 1)
        print(f"[PROFILE chr{chrom}] iter {itr:4d} | "
            f"β {timer_b:6.3f}s  ψ {timer_ps:6.3f}s  other {timer_lo-timer_b-timer_ps:6.3f}s "
            f"(tot {timer_lo:6.3f}s)")

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


