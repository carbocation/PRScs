#!/usr/bin/env python3

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""

import gigrnd

import numpy as np
from scipy import linalg
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from typing import Dict, List, Tuple, Sequence

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

# ─────────────────────── numeric safety guards ────────────────────────
PSI_MIN   = 1e-8
PSI_MAX   = 1e8
SIGMA_MIN = 1e-8

# ---------- helper for one LD block ----------------------------------
def _sample_block_fused(state, psi_slice, sigma, n):
    """
    Draw β for one LD block *and* return (β_block, quadratic form, did_fallback).

    Matches the current `block_state` layout:
        ld_blk, beta_mrg, rng, L, diag_curr, invpsi, zbuf, work
    """
    # ─ aliases --------------------------------------------------------
    L        = state["L"]          # (m×m) lower-tri Cholesky – modified in-place
    diag_o   = state["diag_curr"]  # cached invψ (length-m)
    beta_mrg = state["beta_mrg"]   # marginal β̂  (length-m)
    invpsi   = state["invpsi"]     # scratch (length-m)
    zbuf     = state["zbuf"]       # scratch (length-m)
    rng      = state["rng"]

    # --- Initialize did_fallback ---
    did_fallback = False # Assume fast path initially

    # 0 ─── robust ψ sanitisation (handles NaN/Inf before the reciprocal)
    # This sanitization should ideally use the PSI_MIN/PSI_MAX consistent with your psi update logic
    # If you switched to psi_update_scalar where psi_max is 1.0, this PSI_MAX here might be too large,
    # but psi_slice comes from the global psi array which would have been clipped already.
    # The key is consistency in what psi_slice is expected to be.
    np.nan_to_num(psi_slice,
                  copy=False,
                  nan=1.0, # Or a value consistent with your psi bounds, e.g., median of expected psi
                  posinf=PSI_MAX, # Use the actual upper bound psi_slice could have
                  neginf=PSI_MIN)
    np.clip(psi_slice, PSI_MIN, PSI_MAX, out=psi_slice) # Ensure psi_slice adheres to expected bounds

    # 1 ─── z  ←  N(0, σ/n · I)
    rng.standard_normal(out=zbuf)
    zbuf *= (sigma / n) ** 0.5

    # 2 ─── incremental diagonal update of the cached Cholesky factor
    np.reciprocal(psi_slice, out=invpsi)                 # invψₙₑw
    try:
        # Assuming you have made gigrnd.chol_diag_update_safe_nb more robust
        # (e.g., fastmath=False, internal c_==0 checks returning 1)
        unsafe = gigrnd.chol_diag_update_safe_nb(L, diag_o, invpsi)
    except ZeroDivisionError:
        # This catch block is a good fallback if chol_diag_update_safe_nb itself isn't
        # modified to always return 1 instead of raising ZeroDivisionError.
        # Ideally, chol_diag_update_safe_nb is modified to not throw this.
        log.warning("ZeroDivisionError caught during chol_diag_update_safe_nb. Forcing fallback.")
        unsafe = 1

    if unsafe or not np.isfinite(L).all():
        # --- Fallback path taken ---
        did_fallback = True
        # log.debug(f"Block (ID/index if available) falling back to full Cholesky. unsafe_flag={unsafe}, L_finite={np.isfinite(L).all()}") # Optional debug log
        # Fallback: full (and robust) Cholesky rebuild
        try:
            A = state["ld_blk"] + np.diag(invpsi) # invpsi here is 1/psi_slice
            L[:] = linalg.cholesky(A, lower=True, check_finite=False)
            diag_o[:] = invpsi                     # keep cache coherent
        except linalg.LinAlgError:
            # Log which block failed if possible (would need block index passed to state or similar)
            log.error("Cholesky rebuild failed during fallback – zeroing this block.")
            # Still return three values, including did_fallback status
            return np.zeros_like(beta_mrg), 0.0, did_fallback # did_fallback is true here
    else:
        # --- Fast path succeeded ---
        # did_fallback remains False
        diag_o[:] = invpsi                         # cache stays coherent

    # 3 ─── y = Lᵀ⁻¹ β̂
    try:
        y = linalg.solve_triangular(L,
                                    beta_mrg,
                                    lower=True,
                                    trans='T',
                                    check_finite=False) # Set check_finite=True if L can have NaNs not caught before
    except linalg.LinAlgError as e:
        log.error(f"LinAlgError during first solve_triangular (L.T^-1 beta_mrg): {e}. Zeroing block. Fallback status: {did_fallback}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback
    except ValueError as e: # Can happen if L contains NaNs/infs and check_finite=False
        log.error(f"ValueError during first solve_triangular (L.T^-1 beta_mrg): {e}. Likely NaNs in L. Zeroing block. Fallback status: {did_fallback}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback


    # 4 ─── β = L⁻¹ (y + z)
    try:
        beta_b = linalg.solve_triangular(L,
                                         y + zbuf, # y or zbuf could be non-finite if previous steps had issues
                                         lower=True,
                                         trans='N',
                                         check_finite=False) # Set check_finite=True if y+zbuf could be issues
    except linalg.LinAlgError as e:
        log.error(f"LinAlgError during second solve_triangular (L^-1 (y+z)): {e}. Zeroing block. Fallback status: {did_fallback}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback
    except ValueError as e:
        log.error(f"ValueError during second solve_triangular (L^-1 (y+z)): {e}. Likely NaNs. Zeroing block. Fallback status: {did_fallback}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback

    # Check for NaNs in beta_b before quadratic form calculation
    if not np.isfinite(beta_b).all():
        log.warning(f"Non-finite values in beta_b after solves. Zeroing block. Fallback status: {did_fallback}")
        # Log first few elements of L, y, zbuf, beta_b if this occurs often
        # log.debug(f"L[:3,:3]=\n{L[:3,:3]}")
        # log.debug(f"y[:5]={y[:5]}")
        # log.debug(f"zbuf[:5]={zbuf[:5]}")
        # log.debug(f"beta_b[:5]={beta_b[:5]}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback


    # 5 ─── quadratic form  βᵀ (L Lᵀ) β   ==  ||Lᵀ β||²
    # This can also fail if beta_b is non-finite
    try:
        Lt_beta = L.T @ beta_b
        quad_b  = float(Lt_beta @ Lt_beta)
        if not np.isfinite(quad_b):
            log.warning(f"Non-finite quad_b ({quad_b}). Zeroing block. Fallback status: {did_fallback}")
            # log.debug(f"L.T[:3,:3]=\n{L.T[:3,:3]}")
            # log.debug(f"beta_b[:5]={beta_b[:5]}")
            # log.debug(f"Lt_beta[:5]={Lt_beta[:5]}")
            return np.zeros_like(beta_mrg), 0.0, did_fallback

    except Exception as e: # Catch any other error during quadratic form
        log.error(f"Error during quadratic form calculation: {e}. Zeroing block. Fallback status: {did_fallback}")
        return np.zeros_like(beta_mrg), 0.0, did_fallback


    return beta_b.copy(), quad_b, did_fallback

STATE: List[Dict] | None = None          # one global per process
def _init_worker(shared_state: List[Dict]):
    """Executed once in every child process."""
    global STATE
    STATE = shared_state

def _loky_worker_body(k: int, psi_slice: np.ndarray, sigma: float, n: int):
    """Small wrapper that pulls per-block state from the process-global."""
    return _sample_block_fused(STATE[k], psi_slice, sigma, n)

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')

    n_jobs = int(os.environ.get("PRSCS_N_JOBS", "1"))  # default: 1
    backend = os.getenv("PRSCS_BACKEND", "threading")  # default to threads

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
    active   = [(k, r) for k, r in enumerate(idx_ranges) if blk_size[k]]

    n_jobs = min(n_jobs, len(active))                  # Cap n_jobs at the number of non-empty blocks

    # helper: always return a slice view, never a fancy-index copy
    def _psi_view(r: range) -> np.ndarray:
        return psi_1d[r.start : r.stop]          # shares memory with ψ

    # --- persistent Cholesky state for every LD block -----------------
    block_state: List[Dict | None] = []
    for k, r in enumerate(idx_ranges):
        m = blk_size[k]
        if m == 0:                     # empty block
            block_state.append(None)
            continue

        # keep factor in float64 for numerical headroom
        ld_sub = ld_blk[k].astype(np.float64, copy=False)

        # Target diagonal based on initial psi=1.0
        # invpsi values are expected to be positive. psi is clipped to PSI_MIN, PSI_MAX.
        # So invpsi will be between 1/PSI_MAX and 1/PSI_MIN.
        initial_invpsi_values = np.ones(m, dtype=np.float64) # psi is initialized to 1.0
        try:
            matrix_for_L0 = ld_sub + np.diag(initial_invpsi_values)
            L0 = linalg.cholesky(matrix_for_L0, lower=True, check_finite=True).copy(order='F')
            # diag_curr should store the diagonal that L0 is based on
            current_diag_for_L0 = initial_invpsi_values.copy()
        except linalg.LinAlgError:
            log.warning(f"Initial Cholesky failed for block {k}. Attempting with jitter.")
            jitter_val = 1e-6 # Small absolute jitter
            try:
                # Add jitter to the diagonal components that were summed with ld_sub
                matrix_for_L0_jittered = ld_sub + np.diag(initial_invpsi_values + jitter_val)
                L0 = linalg.cholesky(matrix_for_L0_jittered, lower=True, check_finite=True).copy(order='F')
                current_diag_for_L0 = initial_invpsi_values + jitter_val # L0 is based on this
                log.info(f"Initial Cholesky for block {k} succeeded with jitter {jitter_val}.")
            except linalg.LinAlgError:
                log.error(f"Initial Cholesky for block {k} failed even with jitter. "
                        f"Using identity matrix for L0 as a last resort. Results for this block will be impacted.")
                # Fallback to identity matrix for L0. This will make beta_b likely small or zero after solves.
                L0 = np.eye(m, dtype=np.float64)
                current_diag_for_L0 = np.ones(m, dtype=np.float64) # L0 is identity, so effectively diag(1) was added to 0 matrix

        # L0 = linalg.cholesky(ld_sub + np.eye(m), lower=True).copy(order='F')

        block_state.append(
            dict(
                ld_blk     = ld_sub,
                beta_mrg   = beta_mrg[r].ravel(),                 # view
                rng        = np.random.default_rng(
                                None if seed is None else seed + k
                             ),
                L          = L0,
                diag_curr  = current_diag_for_L0,
                invpsi     = np.empty(m, dtype=np.float64),
                zbuf       = np.empty(m, dtype=np.float64),
                work       = np.empty(m, dtype=np.float64),
            )
        )
    
    # ----------- choose backend --------------------------------------------
    if backend == "threading":
        blas_threads = 0
        workers = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")
        def _submit(k, r):
            return delayed(_sample_block_fused)(
                block_state[k],                # per-block state
                _psi_view(r),                  # ← view, not copy
                sigma, n
            )
    elif backend == "loky":
        blas_threads = 1
        workers = Parallel(
            n_jobs      = n_jobs,
            backend     = "loky",
            initializer = _init_worker,
            initargs    = (block_state,),
        )
        def _submit(k, r):
            return delayed(_loky_worker_body)(
                k,
                _psi_view(r),                  # ← view, not copy
                sigma, n
            )
    else:
        raise ValueError("backend must be 'threading' or 'loky'")

    # one-off banner
    print(f"[DBG] backend={backend}  n_jobs={n_jobs}  active_blocks={len(active)}")

    # ----------- global arrays ---------------------------------------------
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

    beta_est  = np.zeros_like(beta)
    psi_est   = np.zeros_like(psi)
    sigma_est = 0.0
    phi_est = 0.0
    
    # MCMC
    pp = 0
    timer  = collections.Counter()
    counts = collections.Counter()

    for itr in range(1,n_iter+1):
        loop_start = time.perf_counter()
        # -------- β-step: threaded executor, fused update --------------
        t0 = time.perf_counter()
        if n_jobs == 1 or len(active) < 2:
            # serial fast-path
            results = [
                _sample_block_fused(block_state[k],
                                    _psi_view(r),          # ← view
                                    sigma, n)
                for k, r in active
            ]
        else:
            with threadpool_limits(limits=blas_threads, user_api="blas"):
                results = workers(_submit(k, r) for k, r in active)
        timer['beta'] += time.perf_counter() - t0
        counts['beta'] += 1

        # unpack results ------------------------------------------------
        quad = 0.0
        fallbacks_this_iter = 0
        for res_idx, ((k_active, r_active), res_tuple) in enumerate(zip(active, results)):
            beta_b_res, quad_b_res, did_fallback_res = res_tuple # Unpack
            if did_fallback_res:
                fallbacks_this_iter += 1
            if not np.isfinite(beta_b_res).all() or not np.isfinite(quad_b_res):
                raise ValueError(f"Worker returned non-finite beta_b or quad_b for active block index {k_active} (range {r_active}). "
                            f"Replacing with zeros for main accumulation. beta_b_res: {beta_b_res[:5]}, quad_b_res: {quad_b_res}")
                # beta_1d[r_active] = 0.0 # Assign zeros to the corresponding slice in global beta
                # quad_b_res from a block with bad beta_b_res is also suspect, so don't add it or add 0.
                # The 0.0 from _sample_block_fused's own check should propagate here if it triggered.
                # If it became non-finite during transfer or some other reason, quad_b_res could be NaN.
                # if np.isfinite(quad_b_res):
                #     quad += 0.0 # Effectively not adding if beta was bad.
                # else quad remains unchanged, implicitly adding 0 for this bad block's contribution to quad.
            else:
                beta_1d[r_active] = beta_b_res
                quad += quad_b_res
        
        active_block_count = len(active) if active else 0 # Handle case of no active blocks
        log.info(f"[ITER {itr}] Cholesky fallbacks: {fallbacks_this_iter} / {active_block_count} active blocks")

        s1 = float((beta * beta_mrg).sum())
        if not np.isfinite(s1):
            raise FloatingPointError("non-finite s1 – β or ψ corrupted")
        s2 = float((beta**2 / psi).sum())
        if not np.isfinite(s2):
            raise FloatingPointError("non-finite s2 – β or ψ corrupted")

        e1 = float(n/2.0*(1.0 - 2.0*s1 + quad))
        e2 = float(n/2.0*s2)
        err = max(e1, e2)

        if not np.isfinite(err) or err <= 0.0:
            raise FloatingPointError("non-finite or non-positive err in σ step")

        # σ-step (robust scale)
        scale = 1.0 / max(err, 1.0 / SIGMA_MIN)
        # force sigma to be a Python float (not a 0-d array)
        sigma = float(1.0 / np.random.gamma((n + p) / 2.0, scale))

        log.info(f"[ITER {itr}] Sampled sigma: {sigma:.4e}")

        # if not np.isfinite(beta).all():
        #     # full reset of the offending block(s)
        #     beta[np.isnan(beta) | np.isinf(beta)] = 0.0

        # ---------- ψ & δ  fused update  ----------
        t0 = time.perf_counter()
        delta_sum = gigrnd.psi_update_scalar(
            psi_1d, a, b, phi, beta_1d, sigma, n
        )
        # delta_sum = gigrnd.psi_update_fused(
        #     psi_1d,          # ψ is updated in-place
        #     a, b,
        #     phi,
        #     beta_1d,
        #     sigma,
        #     n
        # )
        np.nan_to_num(psi_1d, copy=False, nan=1.0, posinf=PSI_MAX, neginf=PSI_MIN)
        np.clip(psi_1d, PSI_MIN, PSI_MAX, out=psi_1d)        # keeps 1e-8 ≤ ψ ≤ 1e8

        # PSI LOGGING
        # ***** LOG PSI_1D STATISTICS RIGHT HERE *****
        psi_min_val = np.min(psi_1d)
        # If you used psi_update_scalar with a cap of 1.0, PSI_MAX here (1e8) might be misleading for the upper bound check.
        # Adjust the max_bound_check_value accordingly.
        # Assuming PSI_MAX is still 1e8 for this example, but if your effective cap is 1.0, use 1.0.
        max_bound_check_value = 1.0 if False else PSI_MAX
        psi_max_val = np.max(psi_1d)
        psi_mean_val = np.mean(psi_1d)
        # PSI_MIN is defined in your code (likely 1e-8)
        num_at_min_bound = np.sum(psi_1d <= (PSI_MIN + 1e-12)) # Epsilon for float comparison
        # Check against the *actual* upper bound being enforced
        num_at_max_bound = np.sum(psi_1d >= (max_bound_check_value - 1e-9)) # Epsilon for float comparison
        log.info(f"[ITER {itr}] psi_1d stats: min={psi_min_val:.4e}, max={psi_max_val:.4e}, mean={psi_mean_val:.4e}, "
                f"count_at_min_bound={num_at_min_bound}, count_at_max_bound={num_at_max_bound} (upper_bound_checked={max_bound_check_value:.1e})")
        # /PSI LOGGING

        timer['psi'] += time.perf_counter() - t0
        counts['psi'] += 1
        # ------------------------------

        # (optional) φ-step
        if phi_updt == True:
            w = np.random.gamma(1.0, 1.0/(phi+1.0))
            phi = np.random.gamma(p*b + 0.5, 1.0/(delta_sum + w))

        # posterior means / samples
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1
        
        # --- profiling banner ----
        timer['loop'] += time.perf_counter() - loop_start
        counts['loop'] += 1        # same as number of iterations
        timer_b  = timer['beta'] / max(counts['beta'], 1)
        timer_ps = timer['psi']  / max(counts['psi'],  1)
        timer_lo = timer['loop'] / max(counts['loop'], 1)
        print(f"[PROFILE chr{chrom}] iter {itr:4d} | "
            f"β {timer_b:6.3f}s  ψ {timer_ps:6.3f}s  other {timer_lo-timer_b-timer_ps:6.3f}s "
            f"(tot {timer_lo:6.3f}s). Currently, sigma=({sigma_est:.3e}), phi=({phi_est:.3e})")

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
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, float(beta.item())))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, float(psi.item())))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')


