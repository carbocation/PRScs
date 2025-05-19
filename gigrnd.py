#!/usr/bin/env python3

"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.

"""


import math
import numpy as np
from numba import njit, prange

# ─────────────────────────────── constants ──────────────────────────────
# These bounds are safe for IEEE‑754 double precision and do not affect the
# practical posterior mass of the continuous‑shrinkage prior.
PSI_MIN: float = 1.0e-8   # lower bound for every ψ‑draw (≫ 0)
PSI_MAX: float = 1.0e8    # upper bound for every ψ‑draw (≪ exp(709))

# Guard rails for the Cholesky rank‑1 update/downdate functions.
GIG_MIN: float = 1.0e-300        # avoids divide‑by‑zero inside gigrnd
GIG_MAX: float = 1.0e150         # avoids overflow  inside gigrnd
DELTA_VALUE_MAX: float = 1.0e4   # refuse absurdly large updates
L_PIVOT_MIN: float = 1.0e-10     # absolute pivot floor (≈ 10×ε for 64‑bit)
REL_TOL: float = 1.0e-4          # reject downdates that crush ≥99.99 % of pivot
# ────────────────────── helper functions for gigrnd ─────────────────────


@njit(cache=True)
def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f

@njit(cache=True)
def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f

@njit(cache=True)
def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f

@njit(cache=True)
def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    # p = float(p); a = float(a); b = float(b)
    # NOTE: p, a, b must already be Python floats (not arrays)
    lam = p
    omega = math.sqrt(a*b)

    omega2 = omega*omega
    lam2   = lam*lam

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(omega2+lam2)-lam
    alpha2 = alpha*alpha

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0/(alpha+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0/(alpha+2.0*lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0/(alpha*math.cosh(1)+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        elif lam == 0:
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/alpha2+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/alpha2+2.0/alpha)))

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0/xi
    r = 1.0/zeta

    td = t-r*eta
    sd = s-p*theta
    q = td+sd

    # random variate generation
    while True:
        U = np.random.random()
        V = np.random.random()
        W = np.random.random()
        if U < q/(p+q+r):
            rnd = -sd+q*V
        elif U < (q+r)/(p+q+r):
            rnd = td-r*math.log(V)
        else:
            rnd = -sd+p*math.log(V)

        f1 = math.exp(-eta-zeta*(rnd-t))
        f2 = math.exp(-theta+xi*(rnd+s))
        if W*g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+lam2/omega2))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd

@njit(fastmath=False, cache=True)
def psi_update_fused(
    psi: np.ndarray,  # (p,)
    a_hyper: float,
    b_hyper: float,
    phi: float,
    beta: np.ndarray,  # (p,)
    sigma: float,
    n_gwas: int,
) -> float:
    """Simultaneous one-pass update of δ *and* ψ vectors.

    Returns
    -------
    float
        Σ δ_j - the sum of fresh *δ* draws (needed elsewhere in PRS-CS).
    """
    p: int = psi.size
    delta_sum: float = 0.0
    a_minus_half: float = a_hyper - 0.5

    for j in prange(p):
        # ─── 1. draw δ_j ∼ Gamma(a+b, 1 / (ψ + φ)) ───────────────────
        delta_j: float = np.random.gamma(a_hyper + b_hyper, 1.0 / (psi[j] + phi))

        # ─── 2. draw ψ_j ∼ GIG(a−½, 2δ, nβ²/σ) with numeric guards ────
        a_gig: float = min(max(2.0 * delta_j, GIG_MIN), GIG_MAX)
        b_gig: float = min(max(n_gwas * beta[j] * beta[j] / sigma, GIG_MIN), GIG_MAX)

        psi_j: float = gigrnd(a_minus_half, a_gig, b_gig)

        # clip to representable interval (principled numeric guard)
        if not math.isfinite(psi_j):
            psi_j = 1.0     # ultra‑rare fallback
        if psi_j < PSI_MIN:
            psi_j = PSI_MIN
        elif psi_j > PSI_MAX:
            psi_j = PSI_MAX

        psi[j] = psi_j
        delta_sum += delta_j

    return delta_sum

@njit(cache=True, fastmath=False)
def _chol_rank1_inplace(L: np.ndarray, x: np.ndarray, sign: float) -> int:  # noqa: N803
    m: int = x.size
    for k in range(m):
        Lkk: float = L[k, k]
        if Lkk <= L_PIVOT_MIN:
            return 1

        xk: float = x[k]

        if sign < 0.0 and abs(xk) >= Lkk:
            return 1
        
        # Original calculation of r_
        # r_ = math.sqrt(Lkk * Lkk + sign * xk * xk)
        # If Lkk*Lkk + sign*xk*xk is negative (e.g. downdate abs(xk)>Lkk and guard failed)
        # math.sqrt would raise ValueError. Let's be more explicit for stability.
        val_insidesqrt = Lkk * Lkk + sign * xk * xk
        if val_insidesqrt < 0.0:
             # This should ideally be caught by "abs(xk) >= Lkk" for downdates if Lkk > 0
             # unless floating point artifacts with fastmath occur.
            return 1 # Unstable: attempt to take sqrt of negative

        r_: float = math.sqrt(val_insidesqrt)

        # Lkk is guaranteed > L_PIVOT_MIN here.
        # If r_ is 0, it means val_insidesqrt was 0.
        # This implies Lkk*Lkk = -sign*xk*xk.
        # If sign=1, Lkk=0 and xk=0 (Lkk=0 caught by L_PIVOT_MIN).
        # If sign=-1, Lkk*Lkk = xk*xk, so abs(Lkk)=abs(xk). This should be caught by the abs(xk)>=Lkk guard.
        # If it wasn't caught, and r_ is 0, then c_ will be 0.
        if r_ == 0.0 and Lkk > L_PIVOT_MIN : # Check Lkk > L_PIVOT_MIN again for strictness, though implied
            # If r_ is 0 and Lkk is not, c_ will be 0.
            # This situation implies an issue not caught by previous guards, possibly due to fastmath.
            # If any x[j] for j > k is non-zero, or if Ljk_old is non-zero and s_ is non-zero,
            # then division by c_=0 will occur if the k+1 < m block is entered.
            # To be safe, if r_ == 0 and Lkk !=0 (which means c_ will be 0), treat as unstable if c_ is to be used as divisor.
            # The only case where c_=0 might be "fine" is if the k+1 block is not entered or x[j] and Ljk_old make the numerator zero.
            # It's safer to consider r_=0 (and Lkk !=0) as unstable here.
            if k + 1 < m: # If c_ would be used as a divisor
                return 1 # Unstable, c_ would be zero

        # Fallback for Lkk being exactly zero if it somehow passed L_PIVOT_MIN (unlikely)
        if Lkk == 0.0: # This path should not be taken given L_PIVOT_MIN check
             return 1

        c_: float = r_ / Lkk
        s_: float = xk / Lkk
        
        L[k, k] = r_
        if k + 1 < m:
            if c_ == 0.0: # Final explicit check before division
                return 1 # Division by zero would occur
            for j in range(k + 1, m):
                Ljk_old: float = L[j, k]
                L[j, k] = (Ljk_old + sign * s_ * x[j]) / c_
                x[j] = c_ * x[j] - s_ * Ljk_old
    return 0

@njit(cache=True, fastmath=False)
def chol_diag_update_safe_nb(
    L: np.ndarray,
    diag_curr: np.ndarray,
    invpsi_new: np.ndarray,
) -> int:
    """Incrementally apply  Δ = invψ_new − invψ_old to cached *L*.

    Returns 0 if the update was applied in‑place, 1 if the caller must fall
    back to a full Cholesky rebuild.
    """
    m: int = diag_curr.size
    x: np.ndarray = np.empty(m, dtype=np.float64)

    for i in range(m):
        delta_val: float = invpsi_new[i] - diag_curr[i]
        if delta_val == 0.0:
            continue

        # ─── safety checks on the diagonal move ────────────────────
        if delta_val < 0.0:
            new_diag: float = diag_curr[i] + delta_val
            if new_diag <= L_PIVOT_MIN or new_diag <= REL_TOL * diag_curr[i]:
                return 1
        elif delta_val > DELTA_VALUE_MAX:
            return 1

        # ─── apply the rank‑1 update/downdate ──────────────────────
        sign: float = 1.0
        if delta_val < 0.0:
            sign, delta_val = -1.0, -delta_val

        x.fill(0.0)
        x[i] = math.sqrt(delta_val)
        if _chol_rank1_inplace(L, x, sign) != 0:
            return 1

        diag_curr[i] = invpsi_new[i]

    return 0
