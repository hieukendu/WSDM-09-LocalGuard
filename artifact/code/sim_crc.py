"""Numeric check of Proposition 1 on synthetic users (labelled synthetic; not used in the paper's tables).

Each synthetic user has P records with a necessity score, a noisy sensitivity score and a hidden
audited label. The gate admits the K most necessary records whose sensitivity score is at most
lambda (with backfill, so raw exposure is not monotone and the running maximum is used).
We verify E[exposure of a fresh user at lambda_hat] <= alpha over many calibration draws, for
several calibration sizes, noise levels and base rates, including an adversarial case in which the
sensitivity score is anti-correlated with the audited label.
"""
import json
import os

import numpy as np

from crc import crc_index, monotonize

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
K, P = 5, 10
GRID = np.concatenate([[-1.0], np.linspace(0.05, 1.0, 20)])


def users(rng, n, base, noise, sign):
    a = rng.random((n, P)) < base * rng.gamma(2.0, 0.5, size=(n, 1))       # audited labels, user heterogeneity
    s = np.clip(0.5 + sign * 0.25 * (2 * a - 1) + noise * rng.standard_normal((n, P)), 0, 1)
    g = rng.standard_normal((n, P)) + 0.5 * a                             # sensitive records slightly more useful
    order = np.argsort(-g, axis=1)
    a_o, s_o = np.take_along_axis(a, order, 1), np.take_along_axis(s, order, 1)
    L = np.zeros((n, len(GRID)))
    for j, lam in enumerate(GRID):
        keep = s_o <= lam
        rank = np.cumsum(keep, axis=1)
        L[:, j] = (a_o & keep & (rank <= K)).sum(axis=1)
    return L


def main():
    rng = np.random.default_rng(0)
    out = []
    for n in (50, 100, 200, 400):
        for base, noise, sign in ((0.1, 0.15, 1), (0.3, 0.15, 1), (0.3, 0.4, 1), (0.3, 0.15, -1)):
            for alpha in (0.25, 0.5, 1.0):
                T = 4000
                real, mono = [], []
                for _ in range(T):
                    L = users(rng, n + 1, base, noise, sign)
                    Lm = monotonize(L)
                    j = crc_index(Lm[:n], alpha, K)
                    real.append(L[n, j])
                    mono.append(Lm[n, j])
                out.append(dict(n=n, base=base, noise=noise, sign=sign, alpha=alpha,
                                realized=float(np.mean(real)), realized_se=float(np.std(real) / np.sqrt(T)),
                                monotone_loss=float(np.mean(mono)),
                                violated=bool(np.mean(mono) - 3 * np.std(mono) / np.sqrt(T) > alpha)))
                print(out[-1], flush=True)
    json.dump(out, open(os.path.join(RES, "sim_crc.json"), "w"), indent=1)
    print("max (monotone loss - alpha):", max(r["monotone_loss"] - r["alpha"] for r in out),
          "violations:", sum(r["violated"] for r in out))


if __name__ == "__main__":
    main()
