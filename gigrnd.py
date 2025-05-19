#!/usr/bin/env python3

"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.

"""


import math
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f

@njit(cache=True, fastmath=True)
def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f

@njit(cache=True, fastmath=True)
def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f

@njit(cache=True, fastmath=True)
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

@njit(parallel=True, fastmath=True, cache=True)
def psi_update_fused(psi, a, b, phi, beta, sigma, n):
    """
    In-place update of ψ *and* latent δ in a single pass.

        δ_j  ~  Ga(a + b,   1 / (ψ_j + φ))
        ψ_j' ~  GIG(a − ½,  2 δ_j,   n·β_j² / σ)

    Returns
    -------
    delta_sum : float
        ∑_j δ_j  — needed for the global-φ Gibbs step.

    Notes
    -----
    * Keeps exactly the same Gibbs conditionals as PRS-CS.
    * Eliminates the extra δ-array and the second loop over p SNPs.
    """
    p = psi.size
    delta_sum = 0.0
    a_minus_half = a - 0.5

    for j in prange(p):
        # ── draw δ_j ─────────────────────────────────────────
        scale = 1.0 / (psi[j] + phi)            # 1 / (ψ + φ)
        delta_j = np.random.gamma(a + b, scale)

        # ── draw ψ_j using the Devroye sampler ──────────────
        psi_j = gigrnd(a_minus_half,
                       2.0 * delta_j,
                       n * (beta[j] * beta[j]) / sigma)

        # soft upper-clip (matches original code)
        if psi_j > 1.0:
            psi_j = 1.0
        psi[j] = psi_j

        delta_sum += delta_j

    return delta_sum

@njit(cache=True, fastmath=True)
def chol_diag_update_safe_nb(L, diag_curr, invpsi_new):
    """
    Seeger rank-1 updates in float64 with an early-exit safety flag.

    Returns
    -------
    unsafe : int   (0 = all good, 1 = pivot became non-positive)
    """
    n = diag_curr.size
    for i in range(n):
        delta = invpsi_new[i] - diag_curr[i]
        if delta == 0.0:
            continue
        new_piv2 = L[i, i] * L[i, i] + delta
        if new_piv2 <= 1e-12 or new_piv2 != new_piv2:   # <=0 or NaN
            return 1                                   # unsafe
        Liinew = new_piv2 ** 0.5
        c = Liinew / L[i, i]
        if i + 1 < n:
            for j in range(i + 1, n):
                L[j, i] /= c
        L[i, i] = Liinew
        diag_curr[i] = invpsi_new[i]
    return 0
