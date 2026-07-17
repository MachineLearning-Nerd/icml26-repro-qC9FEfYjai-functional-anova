#!/usr/bin/env python3
"""Append the non-toy Claim 3 dependence evidence to the Trackio logbook."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "outputs/dependent_support_summary.json").read_text())
PAGE = "Claim 3 — non-rectangular dependent support"


def trackio(*args: str) -> None:
    subprocess.run(["trackio", "logbook", *args], cwd=ROOT, check=True)


table = "\n".join(
    f"| {r['case']} | {r['unique_support']:,} | {r['support_fraction']:.1%} | "
    f"{r['distinct_probability_values']} | {r['total_correlation_nats']:.3f} | "
    f"{r['reconstruction_max_abs_error']:.2e} | {r['r_squared']:.12f} |"
    for r in S["cases"]
)
body = f"""## Non-toy real-data dependence stress test

The initial analytical case had only 27 support states. This upgrade applies the
authors' general `ModelAnalysis` implementation to three qualitatively different
dependent distributions derived from the real UCI Car domain (full Cartesian
support: {S['full_cartesian_support']:,} states). Each case is non-rectangular,
uses non-uniform empirical probabilities, and evaluates a trained RandomForest
output rather than a hand-written threshold.

| Dependence structure | Unique support | Occupancy | Distinct P values | Total correlation (nats) | Max reconstruction error | R² |
|---|---:|---:|---:|---:|---:|---:|
{table}

The constraints cover an inequality, a deterministic equality, and a modular
higher-order relation. Across **{S['total_unique_case_states']:,} case-states**,
all decompositions reconstruct exactly: worst max error
**{S['maximum_reconstruction_abs_error']:.2e}**, weighted L2 error below 1e-18,
and R²=1. The smallest case has {S['minimum_nonrectangular_support']:,} unique
states—16× the original analytical example—and the largest has
{S['maximum_nonrectangular_support']:,}.
"""

trackio("cell", "markdown", "--page", PAGE, "--title", "Three real-data dependence structures", body)
trackio(
    "cell", "figure", "--page", PAGE, "--title", "Large non-rectangular supports",
    "--image", "outputs/dependent_support_cases.png",
    "--raw", "outputs/dependent_support_summary.json",
)
trackio("pin", "--page", PAGE)
trackio(
    "cell", "markdown", "--page", "Conclusion", "--title", "Claim 3 scope upgrade",
    f"Claim 3 now has non-toy evidence on three real-data, non-uniform, non-rectangular distributions "
    f"with {S['minimum_nonrectangular_support']:,}–{S['maximum_nonrectangular_support']:,} unique states. "
    f"All reconstruct exactly (worst error {S['maximum_reconstruction_abs_error']:.2e}, R²=1), "
    "covering inequality, deterministic, and modular higher-order dependence.",
)
print("DEPENDENCE_UPDATE_BUILT")
