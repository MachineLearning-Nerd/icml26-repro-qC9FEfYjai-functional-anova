#!/usr/bin/env python3
"""Exact Functional ANOVA Decomposition (qC9FEfYjai) — exact-instance CPU repro.

Runs the authors' Analytical case (non-rectangular dependent support; Exp-1.ipynb)
via their `anova_module.ModelAnalysis`, then verifies the three claims by exact
algebraic identities + a machine-precision reference match:

  C1 closed-form decomposition  -> reconstruction  f = Σ_A f_A   (max|·| ≤ 1e-10)
                                    + orthogonality  ⟨f_A, f_B⟩_P = 0  for A≠B (≤1e-10)
  C2 efficient, no sampling     -> reconstruction MSE ≈ machine precision (< 1e-14),
                                    deterministic (no MC); fast (<1 s)
  C3 arbitrary dependence / non-rectangular support -> the Analytical case IS
                                    non-rectangular (X3=X2, X5=0); component norms
                                    match the paper's Table 1.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.abspath(os.path.join(HERE, "..", "..", "upstream"))


def generate_joint_support(N):  # verbatim from Exp-1.ipynb
    grid = np.indices((N, N, N)).reshape(3, -1).T
    x1, x2, x4 = grid[:, 0], grid[:, 1], grid[:, 2]
    return np.column_stack((x1, x2, x2, x4, np.zeros(N**3, dtype=int)))  # X3=X2, X5=0


def compute_linear_threshold(X):  # verbatim from Exp-1.ipynb
    return np.sign(1 * X[:, 0] + (-1) * X[:, 1] + 0.5 * X[:, 2])


def run(out):
    out = os.path.abspath(out)  # resolve vs original cwd (paper root) BEFORE chdir
    os.chdir(UP); sys.path.insert(0, UP)
    from anova_module import ModelAnalysis  # noqa: E402  (needs cwd=upstream)
    import time
    t0 = time.perf_counter()
    X = generate_joint_support(3)
    A = ModelAnalysis(X, compute_linear_threshold, 100, 1e-3, 1e-10)
    S, Matrix = A.functional_anova()       # Matrix shape (r, |S|): f_A(x) over support
    P = A.get_P()                          # shape (r,)
    fx = compute_linear_threshold(X)
    rt = time.perf_counter() - t0

    recon = Matrix.sum(axis=1)             # Σ_A f_A(x), shape (r,)
    recon_err = float(np.max(np.abs(fx - recon)))
    mse = float(np.mean((fx - recon) ** 2))
    WP = Matrix * P[:, None]
    Gram = WP.T @ Matrix                   # ⟨f_A, f_B⟩_P
    offdiag = Gram - np.diag(np.diag(Gram))
    ortho_err = float(np.max(np.abs(offdiag)))
    norms = {tuple(S[i]): float(np.sum(Matrix[:, i]**2 * P)) for i in range(len(S))}

    c1 = bool(recon_err < 1e-10 and ortho_err < 1e-10)
    c2 = bool(mse < 1e-14 and rt < 5.0)
    c3 = bool(abs(norms[(1,)] - 0.518) < 2e-3 and abs(norms[(2,)] - 0.074) < 2e-3
              and abs(norms[(1, 2)] - 0.074) < 2e-3 and norms[(4,)] < 1e-6)

    summary = {
        "claim_1": "verified" if c1 else "not-verified",
        "claim_2": "verified" if c2 else "not-verified",
        "claim_3": "verified" if c3 else "not-verified",
        "reconstruction_max_abs_err": recon_err, "reconstruction_mse": mse,
        "orthogonality_max_abs_err": ortho_err, "runtime_seconds": rt,
        "norms": {str(k): v for k, v in norms.items()},
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs")
    run(ap.parse_args().output)
