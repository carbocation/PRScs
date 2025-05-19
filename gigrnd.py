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

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

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
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha)))

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
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+math.pow(lam,2)/math.pow(omega,2)))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd

@njit(parallel=True, fastmath=True, cache=True)
def gig_rvs_vec(out, a_minus_half, delta, beta, sigma, n):
    """
    Fill `out` (1-D float64 array) with GIG draws in parallel.
    Each element uses the scalar `gigrnd` already defined above.
    """
    p = out.size
    for j in prange(p):
        out[j] = gigrnd(
            a_minus_half,
            2.0 * delta[j],
            n * (beta[j] * beta[j]) / sigma
        )

@njit(fastmath=True, cache=True)
def chol_diag_update_inplace(L, diag_old, diag_new):
    """
    In-place update of a lower-triangular Cholesky factor so that

        L Lᵀ = A_old                (on entry)
        L Lᵀ = A_old + diag(δ)      (on exit),  where δ = diag_new – diag_old

    Both diag_old and diag_new are length-n 1-D arrays holding the diagonal
    of A_old and A_new.  Works because each δᵢ eᵢeᵢᵀ is a rank-1 update.
    """
    n = L.shape[0]
    for i in range(n):
        delta = diag_new[i] - diag_old[i]
        if delta == 0.0:
            continue                 # nothing to do for this row/col

        Lii_old = L[i, i]
        Lii_new = (Lii_old**2 + delta) ** 0.5   # r  in the Seeger algorithm
        c = Lii_new / Lii_old                  # cos
        if i + 1 < n:
            for j in range(i + 1, n):          # scale the column below i
                L[j, i] /= c
        L[i, i] = Lii_new
        diag_old[i] = diag_new[i]              # keep book
