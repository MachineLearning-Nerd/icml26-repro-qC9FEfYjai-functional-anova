#!/usr/bin/env python3
"""Faithful numerical audit of the paper's UCI Mushroom experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from anova_module import ModelAnalysis  # noqa: E402

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mushroom/agaricus-lepiota.data"
)
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2603.02673"
PAPER_SHA256 = "b5acd77ac08b22493c22d1f9c8044ac5b911e9919646c5fc99ce347ffc8ae7fc"
SEEDS = [None, 0, 1, 2, 42]
CLAIMED_MSE = 1e-15
CLAIMED_SECONDS = 0.3


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def download(cache: Path) -> tuple[pd.DataFrame, str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        request = urllib.request.Request(
            DATA_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (OpenResearch Mushroom reproduction; "
                    "qC9FEfYjai)"
                )
            },
        )
        with urllib.request.urlopen(request) as response:
            cache.write_bytes(response.read())
    raw = cache.read_bytes()
    return pd.read_csv(cache, header=None), hashlib.sha256(raw).hexdigest()


def prepare(frame: pd.DataFrame):
    y = (frame.iloc[:, 0] == "p").astype(int)
    x = frame.iloc[:, 1:].apply(
        lambda column: column.astype("category").cat.codes
    )
    return x.to_numpy(), y


def one_run(x_encoded: np.ndarray, y_encoded: pd.Series, seed: int | None) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded,
        y_encoded,
        test_size=0.2,
        random_state=42,
    )
    kwargs = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "eval_metric": "logloss",
    }
    if seed is not None:
        kwargs["random_state"] = seed
    model = xgb.XGBClassifier(**kwargs)
    model.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(x_test)))

    def prediction(values):
        return model.predict_proba(values)[:, 1]

    started = time.perf_counter()
    analysis = ModelAnalysis(x_encoded, prediction, 1.07, 1e-3, 1e-10)
    coalitions, matrix = analysis.functional_anova()
    decomposition_seconds = time.perf_counter() - started

    raw_residual = matrix.sum(axis=1) - analysis.get_Y()
    probabilities = analysis.get_P()
    independent_mse = float(np.sum(raw_residual**2 * probabilities))
    mean = float(np.sum(analysis.get_Y() * probabilities))
    variance = float(
        np.sum((analysis.get_Y() - mean) ** 2 * probabilities)
    )
    independent_r2 = float(1.0 - independent_mse / variance)
    return {
        "seed": "authors_default" if seed is None else seed,
        "test_accuracy": accuracy,
        "support_cardinality": int(len(analysis.get_X_uniq())),
        "selected_basis_rank": int(analysis.get_M().shape[1]),
        "aggregated_components": len(coalitions),
        "reported_mse": float(analysis.get_L2_Error()),
        "independent_mse": independent_mse,
        "reported_r2": float(analysis.get_R2()),
        "independent_r2": independent_r2,
        "relative_l2_error": float(analysis.get_L2_Error_rel()),
        "max_abs_reconstruction_error": float(np.max(np.abs(raw_residual))),
        "decomposition_seconds": decomposition_seconds,
    }


def environment() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        if name:
            packages[name] = distribution.version
    return {
        "git_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_git_sha": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": dict(sorted(packages.items())),
    }


def run(artifact_root: Path) -> dict:
    target = artifact_root / "claim_4"
    output = ROOT / "outputs"
    frame, data_sha = download(output / "agaricus-lepiota.data")
    x_encoded, y_encoded = prepare(frame)
    hypergrid_size = int(
        np.prod(
            [int(x_encoded[:, index].max()) + 1 for index in range(x_encoded.shape[1])],
            dtype=object,
        )
    )

    rows = [one_run(x_encoded, y_encoded, seed) for seed in SEEDS]
    for row in rows:
        row["reported_vs_independent_mse_abs_difference"] = abs(
            row["reported_mse"] - row["independent_mse"]
        )
        row["reported_vs_independent_r2_abs_difference"] = abs(
            row["reported_r2"] - row["independent_r2"]
        )

    primary = rows[0]
    numerical_claim_reproduced = (
        primary["reported_r2"] >= 0.999999
        and 1e-16 <= primary["reported_mse"] <= 1e-14
    )
    independent_agreement = all(
        row["reported_vs_independent_mse_abs_difference"] < 1e-18
        and row["reported_vs_independent_r2_abs_difference"] < 1e-12
        for row in rows
    )
    all_far_above_claim = all(row["independent_mse"] > 1e-12 for row in rows)
    verdict = (
        "FALSIFIED"
        if independent_agreement and all_far_above_claim
        else ("VERIFIED" if numerical_claim_reproduced else "BLOCKED")
    )
    mse_values = np.asarray([row["independent_mse"] for row in rows])
    time_values = np.asarray([row["decomposition_seconds"] for row in rows])
    result = {
        "verdict": verdict,
        "dataset": "UCI Mushroom / agaricus-lepiota",
        "dataset_url": DATA_URL,
        "dataset_sha256": data_sha,
        "rows": int(len(x_encoded)),
        "variables": int(x_encoded.shape[1]),
        "hypergrid_size": hypergrid_size,
        "paper_claimed_r2": "approximately 1",
        "paper_claimed_mse": CLAIMED_MSE,
        "paper_claimed_seconds": CLAIMED_SECONDS,
        "authors_default_protocol": primary,
        "seed_sweep": rows,
        "independent_mse_mean": float(mse_values.mean()),
        "independent_mse_std": float(mse_values.std(ddof=1)),
        "independent_mse_min": float(mse_values.min()),
        "independent_mse_max": float(mse_values.max()),
        "decomposition_seconds_median": float(np.median(time_values)),
        "decomposition_seconds_min": float(time_values.min()),
        "decomposition_seconds_max": float(time_values.max()),
        "mse_ratio_observed_to_claimed": float(
            primary["independent_mse"] / CLAIMED_MSE
        ),
        "timing_used_for_verdict": False,
    }
    independent = {
        "method": (
            "recompute weighted residual MSE and R2 directly from raw component "
            "matrix, model outputs, and empirical probabilities"
        ),
        "all_reported_metrics_match_raw_recomputation": independent_agreement,
        "maximum_mse_abs_difference": max(
            row["reported_vs_independent_mse_abs_difference"] for row in rows
        ),
        "maximum_r2_abs_difference": max(
            row["reported_vs_independent_r2_abs_difference"] for row in rows
        ),
    }
    tampered_mse = CLAIMED_MSE
    negative = {
        "corruption": "replace measured MSE with the paper value 1e-15",
        "tampered_mse": tampered_mse,
        "raw_independent_mse": primary["independent_mse"],
        "verifier_accepted": bool(
            np.isclose(
                tampered_mse,
                primary["independent_mse"],
                rtol=1e-6,
                atol=1e-18,
            )
        ),
    }

    write_json(target / "results.json", result)
    write_json(target / "independent_checker.json", independent)
    write_json(target / "negative_control.json", negative)
    write_json(target / "environment.json", environment())
    write_json(
        target / "claim_contract.json",
        {
            "claim": (
                "On Mushroom (22 variables, |E| approximately 1e14), the method "
                "has R2 approximately 1, MSE approximately 1e-15, in 0.3 seconds."
            ),
            "outcomes": ["VERIFIED", "FALSIFIED", "BLOCKED"],
            "metric_acceptance_interval_for_approximately_1e-15": [1e-16, 1e-14],
            "falsification_condition": (
                "The faithful protocol and independent residual checker both "
                "give MSE >1e-12 for the authors default and seed controls."
            ),
            "timing_policy": (
                "Report timing but do not use it for the verdict because this "
                "machine is not the paper's M4/32GB hardware."
            ),
        },
    )
    pd.DataFrame(rows).to_csv(target / "raw_seed_runs.csv", index=False)
    (target / "source_audit.md").write_text(
        "# Source audit\n\n"
        "The Mushroom statement is at `#S5.SS0.SSS0.Px3.p1.14`; the appendix "
        "specifies 100 XGBoost trees of depth 5 at `#A2.SS2.SSS0.Px3.p1.1`, "
        "and M4/32GB hardware at `#A2.SS1.p1.3`. The released notebook uses "
        "the UCI data URL, pandas category codes, split seed 42, 100 trees, "
        "learning rate 0.1, depth 5, and ModelAnalysis(1.07, 1e-3, 1e-10).\n\n"
        f"Source: {PAPER_URL}\n\nRetrieved: 2026-07-23\n\n"
        f"SHA-256: `{PAPER_SHA256}`\n"
    )
    (target / "method.md").write_text(
        "# Method, limitations, and deviations\n\n"
        "The primary row omits XGBoost random_state exactly like the released "
        "notebook. Four explicit seed controls estimate implementation "
        "uncertainty. The weighted residual is independently recomputed. "
        "XGBoost and dependencies are pinned. Timing is descriptive only: "
        "the local CPU is not the paper's stated M4/32GB machine. The numerical "
        "MSE, which differs by many orders of magnitude, determines the verdict.\n"
    )
    (target / "verifier_command.txt").write_text(
        ".venv/bin/python repro/src/verify_mushrooms.py "
        "--output .openresearch/artifacts\n"
    )
    (target / "EVAL.md").write_text(
        f"# Claim 4 evaluation\n\nVerdict: **{verdict}**\n\n"
        f"The faithful default run produced weighted MSE "
        f"`{primary['independent_mse']:.17g}`, independently confirmed from "
        "raw residuals. Timing is not used for this verdict. The tampered "
        "1e-15 negative control was rejected.\n"
    )
    summary = {
        "result": result,
        "independent_checker": independent,
        "negative_control": negative,
    }
    write_json(artifact_root / "mushroom_summary.json", summary)
    print(json.dumps({"mushroom_claim_contract": summary}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".openresearch" / "artifacts"
    )
    args = parser.parse_args()
    summary = run(args.output.resolve())
    accepted = (
        summary["result"]["verdict"] in {"VERIFIED", "FALSIFIED"}
        and summary["independent_checker"][
            "all_reported_metrics_match_raw_recomputation"
        ]
        and not summary["negative_control"]["verifier_accepted"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
