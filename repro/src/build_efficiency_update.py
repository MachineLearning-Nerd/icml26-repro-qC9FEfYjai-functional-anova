#!/usr/bin/env python3
"""Append the paper-scale Claim 2 efficiency evidence to the Trackio logbook."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = json.loads((ROOT / "outputs/car_efficiency_summary.json").read_text())
PAGE = "Claim 2 — efficient, no sampling"


def trackio(*args: str) -> None:
    subprocess.run(["trackio", "logbook", *args], cwd=ROOT, check=True)


body = f"""## Paper-scale efficiency upgrade — official UCI Car experiment

The initial analytical certificate used only 27 support points. This upgrade
runs the authors' official `Exp-2 (car evaluation).ipynb` protocol on the full
UCI Car Evaluation domain: **{SUMMARY['rows']:,} rows**, six categorical
features, four classes, and **{SUMMARY['full_support_size']:,} Cartesian states**.

The closed-form method computed all
**{SUMMARY['exact_output_explanations']:,} class/state explanations** in
**{SUMMARY['exact_total_seconds']:.2f} seconds**. The notebook's sampling-based
KernelSHAP baseline required **{SUMMARY['kernel_total_seconds']:.2f} seconds**
for only **{SUMMARY['kernel_output_explanations']:,} explanations** using its
200-point background set. This is a
**{SUMMARY['throughput_ratio_exact_over_kernel']:.1f}× throughput advantage**
for the exact method, while reconstructing the model probabilities to maximum
absolute error **{SUMMARY['max_reconstruction_abs_error']:.2e}**.

This is not an extrapolation from the tiny analytical example: it is a complete
enumeration of the real dataset's full categorical support, across every output
class, compared directly with the paper's sampling baseline. Dataset SHA-256:
`{SUMMARY['dataset_sha256']}`.
"""

trackio("cell", "markdown", "--page", PAGE, "--title", "Full-scale efficiency certificate", body)
trackio(
    "cell", "figure", "--page", PAGE, "--title", "Exact ANOVA versus KernelSHAP",
    "--image", "outputs/car_efficiency.png", "--raw", "outputs/car_efficiency_summary.json",
)
trackio("pin", "--page", PAGE)
print("EFFICIENCY_UPDATE_BUILT")
