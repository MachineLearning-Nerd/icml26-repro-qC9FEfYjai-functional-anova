#!/usr/bin/env python3
"""Paper-scale efficiency reproduction on the official Car Evaluation example.

This is a script version of ``Exp-2 (car evaluation).ipynb``.  It compares the
paper's closed-form FullSupportAnova decomposition with the sampling-based
KernelSHAP baseline used by the authors.  The exact method explains every point
in the 1,728-state Cartesian support; KernelSHAP is timed on 250 points with the
notebook's 200-point background set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from anova_module import FullSupportAnova, batch_shapley_values  # noqa: E402

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


def load_data(cache: Path) -> tuple[np.ndarray, np.ndarray, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(DATA_URL, cache)
    raw = cache.read_bytes()
    frame = pd.read_csv(cache, names=COLUMNS)
    x = OrdinalEncoder().fit_transform(frame.iloc[:, :-1].values)
    y = LabelEncoder().fit_transform(frame.iloc[:, -1].values)
    return x, y, hashlib.sha256(raw).hexdigest()


def run(output: Path, kernel_points: int) -> dict:
    np.random.seed(0)
    output.mkdir(parents=True, exist_ok=True)
    x, y, data_sha = load_data(output / "car.data")
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1
    ).fit(x, y)

    dims = x.shape[1]
    cardinalities = np.array([int(x[:, j].max()) + 1 for j in range(dims)])
    support_size = int(np.prod(cardinalities))
    probabilities = np.full(support_size, 1.0 / support_size)
    n_classes = len(np.unique(y))
    kernel_points = min(kernel_points, support_size - 1)
    background = x[:200]

    rows = []
    exact_values = []
    support = None
    exact_total = 0.0
    kernel_total = 0.0
    max_reconstruction_error = 0.0

    for class_id in range(n_classes):
        def prediction(z, class_id=class_id):
            return model.predict_proba(z)[:, class_id]

        start = time.perf_counter()
        decomposition = FullSupportAnova(cardinalities, probabilities, prediction)
        subsets, matrix = decomposition.get_anova_full()
        generalized_shapley = batch_shapley_values(dims, subsets, matrix)
        exact_seconds = time.perf_counter() - start
        exact_total += exact_seconds
        if support is None:
            support = decomposition._generate_tuples()
        reconstruction_error = float(
            np.max(np.abs(matrix.sum(axis=1) - prediction(support)))
        )
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
        exact_values.append(generalized_shapley)

        explainer = shap.KernelExplainer(prediction, background)
        start = time.perf_counter()
        kernel_values = np.asarray(
            explainer.shap_values(support[:kernel_points], silent=True)
        )
        kernel_seconds = time.perf_counter() - start
        kernel_total += kernel_seconds

        squared_difference = (
            generalized_shapley[:kernel_points] - kernel_values
        ) ** 2
        rows.append(
            {
                "class_id": class_id,
                "support_size": support_size,
                "kernel_points": kernel_points,
                "exact_seconds_all_support": exact_seconds,
                "kernel_seconds_subset": kernel_seconds,
                "exact_reconstruction_max_abs": reconstruction_error,
                "mean_shapley_squared_difference": float(squared_difference.mean()),
            }
        )

    exact_explanations = support_size * n_classes
    kernel_explanations = kernel_points * n_classes
    exact_throughput = exact_explanations / exact_total
    kernel_throughput = kernel_explanations / kernel_total
    summary = {
        "dataset": "UCI Car Evaluation",
        "dataset_url": DATA_URL,
        "dataset_sha256": data_sha,
        "rows": int(len(x)),
        "features": int(dims),
        "classes": int(n_classes),
        "cardinalities": cardinalities.tolist(),
        "full_support_size": support_size,
        "exact_output_explanations": exact_explanations,
        "kernel_output_explanations": kernel_explanations,
        "kernel_background_points": int(len(background)),
        "exact_total_seconds": exact_total,
        "kernel_total_seconds": kernel_total,
        "exact_explanations_per_second": exact_throughput,
        "kernel_explanations_per_second": kernel_throughput,
        "throughput_ratio_exact_over_kernel": exact_throughput / kernel_throughput,
        "wall_time_ratio_kernel_over_exact": kernel_total / exact_total,
        "max_reconstruction_abs_error": max_reconstruction_error,
        "mean_shapley_squared_difference": float(
            np.mean([r["mean_shapley_squared_difference"] for r in rows])
        ),
        "claim_2_full_scale": "verified" if (
            support_size >= 1000
            and max_reconstruction_error < 1e-10
            and exact_throughput > kernel_throughput
        ) else "not-verified",
    }
    pd.DataFrame(rows).to_csv(output / "car_efficiency_by_class.csv", index=False)
    (output / "car_efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Exact closed form\n(all 1,728 states)", f"KernelSHAP\n({kernel_points} states)"]
    ax.bar(labels, [exact_total, kernel_total], color=["#247ba0", "#f25f5c"])
    ax.set_ylabel("Wall time across four output classes (seconds)")
    ax.set_title(
        f"Exact method: {summary['throughput_ratio_exact_over_kernel']:.0f}× higher explanation throughput"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "car_efficiency.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--kernel-points", type=int, default=250)
    args = parser.parse_args()
    run(args.output.resolve(), args.kernel_points)
