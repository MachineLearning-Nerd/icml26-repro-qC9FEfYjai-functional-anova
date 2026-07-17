#!/usr/bin/env python3
"""Non-toy verification under multiple dependent, non-rectangular supports.

The authors' general ``ModelAnalysis`` implementation is applied to three
distributions derived from the real UCI Car Evaluation domain.  Each removes a
different part of the Cartesian support and uses non-uniform empirical weights.
The constraints exercise inequality, deterministic equality, and modular
higher-order dependence rather than repeating one synthetic construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from anova_module import ModelAnalysis  # noqa: E402


def total_correlation(x: np.ndarray) -> float:
    """Empirical KL(P(X) || product_i P(X_i)), in nats."""
    _, joint_counts = np.unique(x, axis=0, return_counts=True)
    joint_p = joint_counts / joint_counts.sum()
    joint_h = -float(np.sum(joint_p * np.log(joint_p)))
    marginal_h = 0.0
    for j in range(x.shape[1]):
        _, counts = np.unique(x[:, j], return_counts=True)
        p = counts / counts.sum()
        marginal_h -= float(np.sum(p * np.log(p)))
    return marginal_h - joint_h


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(output / "car.data", header=None)
    x = OrdinalEncoder().fit_transform(frame.iloc[:, :-1]).astype(int)
    y = LabelEncoder().fit_transform(frame.iloc[:, -1])
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1
    ).fit(x, y)
    full_support = len(np.unique(x, axis=0))

    base_cases = [
        (
            "ordered_buying_maintenance",
            x[x[:, 0] <= x[:, 1]],
            lambda z: 1 + z[:, 5],
            0,
            "inequality constraint buying <= maintenance; weights 1+safety",
        ),
        (
            "persons_equals_safety",
            x[x[:, 3] == x[:, 5]],
            lambda z: 1 + z[:, 0],
            1,
            "deterministic equality persons = safety; weights 1+buying",
        ),
        (
            "modular_door_constraint",
            x[x[:, 2] == ((x[:, 0] + x[:, 1]) % 4)],
            lambda z: 1 + z[:, 3],
            2,
            "higher-order doors = (buying + maintenance) mod 4; weights 1+persons",
        ),
    ]

    rows = []
    for name, support, weight_fn, class_id, description in base_cases:
        weighted = np.repeat(support, weight_fn(support), axis=0)

        def prediction(z, class_id=class_id):
            return model.predict_proba(z)[:, class_id]

        start = time.perf_counter()
        analysis = ModelAnalysis(weighted, prediction, 100, 1e-3, 1e-10)
        seconds = time.perf_counter() - start
        subsets, matrix = analysis.functional_anova()
        reconstruction = float(
            np.max(np.abs(matrix.sum(axis=1) - analysis.get_Y()))
        )
        probabilities = analysis.get_P()
        unique_support = len(analysis.get_X_uniq())
        rows.append(
            {
                "case": name,
                "description": description,
                "observations_with_weights": int(len(weighted)),
                "unique_support": int(unique_support),
                "full_cartesian_support": int(full_support),
                "support_fraction": unique_support / full_support,
                "removed_states": int(full_support - unique_support),
                "total_correlation_nats": total_correlation(weighted),
                "distinct_probability_values": int(len(np.unique(probabilities))),
                "min_probability": float(probabilities.min()),
                "max_probability": float(probabilities.max()),
                "basis_shape": list(analysis.get_M().shape),
                "components": int(len(subsets)),
                "runtime_seconds": seconds,
                "reconstruction_max_abs_error": reconstruction,
                "weighted_l2_error": float(analysis.get_L2_Error()),
                "r_squared": float(analysis.get_R2()),
            }
        )

    minimum_support = min(row["unique_support"] for row in rows)
    maximum_error = max(row["reconstruction_max_abs_error"] for row in rows)
    minimum_dependence = min(row["total_correlation_nats"] for row in rows)
    summary = {
        "dataset": "UCI Car Evaluation",
        "full_cartesian_support": full_support,
        "cases": rows,
        "number_of_dependence_structures": len(rows),
        "minimum_nonrectangular_support": minimum_support,
        "maximum_nonrectangular_support": max(row["unique_support"] for row in rows),
        "total_unique_case_states": sum(row["unique_support"] for row in rows),
        "maximum_reconstruction_abs_error": maximum_error,
        "minimum_total_correlation_nats": minimum_dependence,
        "all_nonrectangular": all(row["removed_states"] > 0 for row in rows),
        "all_nonuniform": all(row["distinct_probability_values"] > 1 for row in rows),
        "all_exact": all(
            row["reconstruction_max_abs_error"] < 1e-9
            and row["weighted_l2_error"] < 1e-18
            and row["r_squared"] > 0.999999
            for row in rows
        ),
    }
    pd.DataFrame(rows).to_csv(output / "dependent_support_cases.csv", index=False)
    (output / "dependent_support_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    labels = [row["case"].replace("_", "\n") for row in rows]
    ax.bar(labels, [row["unique_support"] for row in rows], color="#70c1b3")
    ax.axhline(full_support, color="#f25f5c", linestyle="--", label="full rectangular support")
    ax.set_ylabel("Unique support states")
    ax.set_title("Three real-data non-rectangular dependence structures")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "dependent_support_cases.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.output.resolve())
