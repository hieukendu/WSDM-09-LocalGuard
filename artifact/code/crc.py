"""Conformal risk control for a monotone admission family (exposure upper bound)."""
import numpy as np


def monotonize(L):
    """Running maximum along the threshold axis: L~(lambda_j) = max_{j' <= j} L(lambda_j')."""
    return np.maximum.accumulate(L, axis=1)


def crc_index(Lcal, alpha, B):
    """Largest grid index j with (n * mean(Lcal[:, j]) + B) / (n + 1) <= alpha.

    Lcal: (n, J+1) calibration losses, nondecreasing in j, column 0 identically zero
    (the empty admission set). Returns 0 when no positive index is feasible.
    """
    n = Lcal.shape[0]
    ok = (n * Lcal.mean(axis=0) + B) / (n + 1) <= alpha
    ok[0] = True
    # losses are monotone, so feasible indices form a prefix
    return int(np.max(np.nonzero(np.logical_and.accumulate(ok))[0]))


def ucb_index(Lcal, alpha, B, delta):
    """High-probability variant: Hoeffding upper confidence bound with a union bound over the grid."""
    n, J = Lcal.shape
    slack = B * np.sqrt(np.log(J / delta) / (2 * n))
    ok = Lcal.mean(axis=0) + slack <= alpha
    ok[0] = True
    return int(np.max(np.nonzero(np.logical_and.accumulate(ok))[0]))


def hb_pvalue(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha, losses in [0, 1] (Bates et al., 2021)."""
    from scipy.stats import binom
    if alpha >= 1:
        return 0.0
    a = min(rhat, alpha)

    def h1(x, y):
        x = min(max(x, 1e-12), 1 - 1e-12)
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))
    p_h = np.exp(-n * h1(a, alpha)) if rhat < alpha else 1.0
    p_b = np.e * binom.cdf(np.ceil(n * rhat), n, alpha)
    return float(min(p_h, p_b, 1.0))


def ltt_index(Lcal, alpha, B, delta):
    """Learn-then-Test with Hoeffding-Bentkus p-values and fixed-sequence testing from the
    smallest parameter upward. With probability at least 1 - delta over the calibration draw,
    the risk of the returned parameter is at most alpha."""
    n, J = Lcal.shape
    j_hat = 0
    for j in range(1, J):
        if hb_pvalue(Lcal[:, j].mean() / B, n, alpha / B) <= delta:
            j_hat = j
        else:
            break
    return j_hat
