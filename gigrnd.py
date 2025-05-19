#!/usr/bin/env python3

"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.

"""


import math
import numpy as np
from numba import njit, prange

GIG_MIN = 1e-300          # > 0   (avoids divide-by-zero)
GIG_MAX = 1e+150          # << exp(709)   (avoids overflow)
DELTA_VALUE_MAX = 1.0e4   # refuse extreme updates

L_PIVOT_MIN = 1e-10      # absolute   (≈ 10 × ε for 64-bit)
REL_TOL     = 1e-4       # relative   (reject delta_value that kills ≥ 99.99 % of a pivot)

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

@njit(fastmath=True, cache=True)
def psi_update_fused(psi, a, b, phi, beta, sigma, n):
    p = psi.size
    delta_sum = 0.0
    a_minus_half = a - 0.5

    for j in prange(p):
        # ── δ_j  ~  Ga(a+b, 1/(ψ+φ)) ────────────────────────────────
        delta_j = np.random.gamma(a + b, 1.0 / (psi[j] + phi))

        # ── prepare *safe* GIG parameters ───────────────────────────
        a_gig = min(max(2.0 * delta_j,               GIG_MIN), GIG_MAX)
        b_gig = min(max(n * beta[j] * beta[j] / sigma, GIG_MIN), GIG_MAX)

        # ── ψ_j  ~  GIG(a−½, a_gig, b_gig) ─────────────────────────
        psi_j = gigrnd(a_minus_half, a_gig, b_gig)

        # final defence: if any arithmetic above still produced a
        # non-finite value, fall back to 1.0 (original PRS-CS clip)
        if not math.isfinite(psi_j) or psi_j > 1.0:
            psi_j = 1.0
        psi[j] = psi_j

        delta_sum += delta_j
    return delta_sum

@njit(cache=True, fastmath=True)
def _chol_rank1_inplace(L, x, sign):
    """
    In-place rank-1 Cholesky update/downdate.
    Returns 0 on success, 1 if the step would make A non-SPD.
    """
    m = x.size
    for k in range(m):
        Lkk = L[k, k]

        # ─── refuse to touch near-zero pivots ──────────────────
        if Lkk <= L_PIVOT_MIN:      # L_PIVOT_MIN
            return 1                # caller will rebuild from scratch

        xk = x[k]
        if sign < 0.0 and abs(xk) >= Lkk:      # classic SPD test
            return 1

        r = math.sqrt(Lkk*Lkk + sign*xk*xk)
        c = r / Lkk
        s = xk / Lkk
        L[k, k] = r
        if k + 1 < m:
            for j in range(k + 1, m):
                Ljk_old = L[j, k]
                L[j, k] = (Ljk_old + sign*s*x[j]) / c
                x[j]    = c*x[j] - s*Ljk_old
    return 0

@njit(cache=True, fastmath=True)
def chol_diag_update_safe_nb(L, diag_curr, invpsi_new):
    """
    Incrementally apply     delta_value = invψ_new - invψ_old
    to the cached Cholesky factor L.

    Returns
    -------
    0 … update succeeded in-place
    1 … step declared unsafe; caller must rebuild the factor

    A step is unsafe if
      • a DOWndate would shrink a pivot below   max(REL_TOL·old, L_PIVOT_MIN)
      • an UPdate would increase a single diagonal by more than DELTA_VALUE_MAX
    """

    m = diag_curr.size
    x = np.empty(m, np.float64)

    for i in range(m):
        delta_value = invpsi_new[i] - diag_curr[i]
        if delta_value == 0.0:
            continue

        # ─── dangerous downdate? ──────────────────────────────
        if delta_value < 0.0:
            new_diag = diag_curr[i] + delta_value
            if (new_diag <= L_PIVOT_MIN or
                new_diag <= REL_TOL * diag_curr[i]):
                return 1

        # ─── dangerous up-date?  ──────────────────────────────
        if delta_value > DELTA_VALUE_MAX:
            return 1

        # rank-1 update/downdate
        sign = 1.0
        if delta_value < 0.0:
            sign, delta_value = -1.0, -delta_value

        x.fill(0.0)
        x[i] = math.sqrt(delta_value)
        if _chol_rank1_inplace(L, x, sign) != 0:      # SPD check
            return 1

        diag_curr[i] = invpsi_new[i]

    return 0
