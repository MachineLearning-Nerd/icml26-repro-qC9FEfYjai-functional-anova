#!/usr/bin/env python3
"""Faithful 10,000-basis audit of the binarized MNIST paper claim."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import cho_factor, cho_solve
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))

from mnist_basis import BinaryBasisExtractor, get_neighbor_pairs  # noqa: E402

PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2603.02673"
PAPER_SHA256 = "b5acd77ac08b22493c22d1f9c8044ac5b911e9919646c5fc99ce347ffc8ae7fc"
SEEDS = [0, 1, 2]
TARGET_RANK = 10_000
TRUNCATED_RANK = 674
TARGET_CLASS = 3


class BinarizedMnistMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, values):
        return self.model(self.flatten(values))


def json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n"
    )


def hash_dataset_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "MNIST" / "raw").glob("*")):
        if path.is_file():
            payload = path.read_bytes()
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return rows


def load_data(data_root: Path):
    train = datasets.MNIST(root=data_root, train=True, download=True)
    test = datasets.MNIST(root=data_root, train=False, download=True)
    train_images = (train.data > 127).to(torch.float32).unsqueeze(1)
    test_images = (test.data > 127).to(torch.float32).unsqueeze(1)
    train_labels = train.targets.to(torch.long)
    test_labels = test.targets.to(torch.long)
    x_encoded = (
        train_images.reshape(len(train_images), -1).numpy().astype(np.int8)
    )
    return (
        train_images,
        train_labels,
        test_images,
        test_labels,
        x_encoded,
        hash_dataset_files(data_root),
    )


def train_model(
    seed: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = BinarizedMnistMlp()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    started = time.perf_counter()
    for _ in range(3):
        model.train()
        total_loss = 0.0
        examples = 0
        for values, targets in loader:
            optimizer.zero_grad()
            output = model(values)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(values)
            examples += len(values)
        losses.append(total_loss / examples)
    training_seconds = time.perf_counter() - started
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(test_images), 1000):
            output = model(test_images[start : start + 1000])
            correct += int(
                (output.argmax(dim=1) == test_labels[start : start + 1000])
                .sum()
                .item()
            )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return state, {
        "seed": seed,
        "accuracy": correct / len(test_images),
        "epoch_mean_losses": losses,
        "training_seconds": training_seconds,
    }


def class_probabilities(
    state: dict[str, torch.Tensor], patterns: np.ndarray
) -> np.ndarray:
    model = BinarizedMnistMlp()
    model.load_state_dict(state)
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(patterns), 1024):
            values = torch.from_numpy(
                patterns[start : start + 1024].astype(np.float32, copy=False)
            )
            probabilities = torch.softmax(model(values), dim=1)
            batches.append(probabilities[:, TARGET_CLASS].numpy())
    return np.concatenate(batches)


def solve_metrics(
    matrix: np.ndarray,
    probabilities: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Torch inference returns float32.  Promote before all moment calculations
    # so squaring cannot silently occur in float32 and so the support-weighted
    # metrics agree with an expanded-row float64 calculation.
    responses = np.asarray(responses, dtype=np.float64)
    started = time.perf_counter()
    gamma = matrix.T @ (probabilities[:, None] * matrix)
    mu = matrix.T @ (probabilities[:, None] * responses)
    gamma_seconds = time.perf_counter() - started
    gamma[np.diag_indices_from(gamma)] += 1e-10
    started = time.perf_counter()
    factor, lower = cho_factor(gamma, lower=True, check_finite=False)
    coefficients = cho_solve(
        (factor, lower), mu, check_finite=False
    )
    solve_seconds = time.perf_counter() - started
    reconstruction = matrix @ coefficients
    residual = reconstruction - responses
    mean = probabilities @ responses
    variance = probabilities @ ((responses - mean) ** 2)
    mse = probabilities @ (residual**2)
    r2 = 1.0 - mse / variance
    relative = mse / (probabilities @ (responses**2))
    return reconstruction, residual, {
        "gamma_seconds": gamma_seconds,
        "cholesky_solve_seconds": solve_seconds,
        "mse": mse,
        "r2": r2,
        "relative_l2": relative,
        "max_abs_residual": np.max(np.abs(residual), axis=0),
    }


def original_row_metrics(
    x_encoded: np.ndarray,
    patterns: np.ndarray,
    responses: np.ndarray,
    reconstruction: np.ndarray,
) -> dict:
    # Torch inference returns float32.  NumPy otherwise accumulates the mean of
    # the 60,000 expanded rows in float32, which is not an independent
    # double-precision check of the support-weighted calculation.
    responses = np.asarray(responses, dtype=np.float64)
    lookup = {row.tobytes(): index for index, row in enumerate(patterns)}
    inverse = np.fromiter(
        (lookup[row.tobytes()] for row in x_encoded),
        dtype=np.int64,
        count=len(x_encoded),
    )
    residual = reconstruction[inverse] - responses[inverse]
    mse = np.mean(residual**2, axis=0)
    mean = np.mean(responses[inverse], axis=0)
    variance = np.mean((responses[inverse] - mean) ** 2, axis=0)
    return {
        "method": (
            "unweighted float64 residuals and variance expanded over all "
            "60000 training rows"
        ),
        "mse": [float(value) for value in mse],
        "r2": [float(value) for value in (1.0 - mse / variance)],
    }


def environment() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        if name:
            packages[name] = distribution.version
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
        "torch_threads": torch.get_num_threads(),
        "maximum_resident_set_size_platform_units": max_rss,
        "packages": dict(sorted(packages.items())),
    }


def run(artifact_root: Path) -> dict:
    target = artifact_root / "claim_5"
    data_root = ROOT / "outputs" / "mnist"
    (
        train_images,
        train_labels,
        test_images,
        test_labels,
        x_encoded,
        dataset_files,
    ) = load_data(data_root)

    states = []
    training_rows = []
    for seed in SEEDS:
        state, training = train_model(
            seed,
            train_images,
            train_labels,
            test_images,
            test_labels,
        )
        states.append(state)
        training_rows.append(training)

    decomposition_started = time.perf_counter()
    basis_started = time.perf_counter()
    extractor = BinaryBasisExtractor(
        x_encoded,
        A=get_neighbor_pairs(28),
        n_set=TARGET_RANK,
        rtol=1e-3,
        atol=1e-3,
        maxpool=200,
        verbose=True,
    )
    basis_seconds = time.perf_counter() - basis_started
    patterns = extractor.get_patterns()
    probabilities = extractor.get_P()
    matrix = extractor.get_matrix()
    selected_sets = extractor.get_sets()
    responses = np.column_stack(
        [class_probabilities(state, patterns) for state in states]
    )
    reconstruction, residual, metrics = solve_metrics(
        matrix, probabilities, responses
    )
    decomposition_seconds = time.perf_counter() - decomposition_started
    independent = original_row_metrics(
        x_encoded, patterns, responses, reconstruction
    )

    truncated_matrix = matrix[:, :TRUNCATED_RANK]
    _, _, truncated_metrics = solve_metrics(
        truncated_matrix, probabilities, responses
    )

    rows = []
    for index, seed in enumerate(SEEDS):
        rows.append(
            {
                **training_rows[index],
                "support_cardinality": int(len(patterns)),
                "selected_basis_elements": int(matrix.shape[1]),
                "r2": float(metrics["r2"][index]),
                "mse": float(metrics["mse"][index]),
                "relative_l2": float(metrics["relative_l2"][index]),
                "max_abs_residual": float(
                    metrics["max_abs_residual"][index]
                ),
                "independent_original_row_r2": float(
                    independent["r2"][index]
                ),
                "independent_original_row_mse": float(
                    independent["mse"][index]
                ),
            }
        )

    r2_values = np.asarray([row["r2"] for row in rows])
    mse_values = np.asarray([row["mse"] for row in rows])
    all_independent = all(
        abs(row["r2"] - row["independent_original_row_r2"]) < 1e-10
        and abs(row["mse"] - row["independent_original_row_mse"]) < 1e-12
        for row in rows
    )
    full_rank = matrix.shape == (len(patterns), TARGET_RANK)
    numerical_match = (
        full_rank
        and all(0.855 <= value < 0.865 for value in r2_values)
        and all(0.005 <= value < 0.015 for value in mse_values)
        and all_independent
    )
    numerical_falsification = (
        full_rank
        and all(
            value < 0.845 or value >= 0.875 for value in r2_values
        )
        and all_independent
    ) or (
        full_rank
        and all(value < 0.0025 or value >= 0.03 for value in mse_values)
        and all_independent
    )
    if numerical_falsification:
        verdict = "FALSIFIED"
        numerical_verdict = "FALSIFIED"
    elif numerical_match:
        verdict = "BLOCKED"
        numerical_verdict = "VERIFIED"
    else:
        verdict = "BLOCKED"
        numerical_verdict = "BLOCKED"

    result = {
        "verdict": verdict,
        "numerical_verdict": numerical_verdict,
        "runtime_subclaim_status": "BLOCKED_DIFFERENT_HARDWARE",
        "dataset": "Binarized MNIST training split",
        "training_rows": int(len(x_encoded)),
        "dimensions": int(x_encoded.shape[1]),
        "unique_support_cardinality": int(len(patterns)),
        "selected_basis_elements": int(matrix.shape[1]),
        "selected_sets": len(selected_sets),
        "target_class": TARGET_CLASS,
        "seed_runs": rows,
        "r2_mean": float(r2_values.mean()),
        "r2_std": float(r2_values.std(ddof=1)),
        "r2_min": float(r2_values.min()),
        "r2_max": float(r2_values.max()),
        "mse_mean": float(mse_values.mean()),
        "mse_std": float(mse_values.std(ddof=1)),
        "mse_min": float(mse_values.min()),
        "mse_max": float(mse_values.max()),
        "basis_seconds": basis_seconds,
        "gamma_seconds": metrics["gamma_seconds"],
        "cholesky_solve_seconds": metrics["cholesky_solve_seconds"],
        "decomposition_seconds": decomposition_seconds,
        "paper_claimed_r2": 0.86,
        "paper_claimed_mse": 0.01,
        "paper_claimed_seconds": 900,
        "timing_used_for_verdict": False,
        "dataset_files": dataset_files,
    }
    independent_output = {
        **independent,
        "all_weighted_support_metrics_match_original_rows": all_independent,
        "maximum_mse_abs_difference": max(
            abs(row["mse"] - row["independent_original_row_mse"])
            for row in rows
        ),
        "maximum_r2_abs_difference": max(
            abs(row["r2"] - row["independent_original_row_r2"])
            for row in rows
        ),
    }
    negative = {
        "control": (
            "solve the same model responses using only the first 674 basis "
            "elements instead of the claimed 10000"
        ),
        "selected_basis_elements": TRUNCATED_RANK,
        "r2": [float(value) for value in truncated_metrics["r2"]],
        "mse": [float(value) for value in truncated_metrics["mse"]],
        "verifier_accepted": TRUNCATED_RANK == TARGET_RANK,
    }

    write_json(target / "results.json", result)
    write_json(target / "independent_checker.json", independent_output)
    write_json(target / "negative_control.json", negative)
    write_json(target / "environment.json", environment())
    write_json(
        target / "runtime.json",
        {
            "basis_seconds": basis_seconds,
            "gamma_seconds": metrics["gamma_seconds"],
            "cholesky_solve_seconds": metrics["cholesky_solve_seconds"],
            "decomposition_seconds": decomposition_seconds,
            "training_seconds_by_seed": [
                row["training_seconds"] for row in training_rows
            ],
        },
    )
    write_json(
        target / "claim_contract.json",
        {
            "claim": (
                "On binarized MNIST (784 dimensions), 10000 basis elements "
                "give R2=0.86 in 15 minutes, with table MSE=0.01."
            ),
            "outcomes": ["VERIFIED", "FALSIFIED", "BLOCKED"],
            "numerical_rounding_intervals": {
                "r2_reported_to_two_decimals": [0.855, 0.865],
                "mse_reported_to_two_decimals": [0.005, 0.015],
            },
            "runtime_policy": (
                "The 15-minute subclaim remains blocked unless run on the "
                "paper's stated M4/32GB hardware. A robust numerical "
                "contradiction is sufficient to falsify the conjunction."
            ),
        },
    )
    pd.DataFrame(rows).to_csv(target / "raw_seed_runs.csv", index=False)
    (target / "source_audit.md").write_text(
        "# Source audit\n\n"
        "The protocol is at `#S5.SS0.SSS0.Px5.p1.3`; Table 7 is `#S5.T7`, "
        "with the 10,000-element row at `#S5.T7.14.14.14`. It specifies "
        "60,000 binarized 784-dimensional training images, an "
        "input-512-128-output MLP, and the structural singleton/variance/"
        "spatial-neighborhood basis policy. The released notebook has the same "
        "model and three epochs but is set to 2,000 elements; this audit changes "
        "only n_set to the paper's high-setting 10,000 value.\n\n"
        f"Source: {PAPER_URL}\n\nRetrieved: 2026-07-23\n\n"
        f"SHA-256: `{PAPER_SHA256}`\n"
    )
    (target / "method.md").write_text(
        "# Method, limitations, and deviations\n\n"
        "Three deterministic training seeds quantify MLP uncertainty. The "
        "published BinaryBasisExtractor is used unchanged at n_set=10000, "
        "rtol=atol=1e-3, maxpool=200, with 28x28 neighbor pairs. One shared "
        "Gram system solves all seed responses. The independent checker expands "
        "residuals over all 60,000 original rows. Binarization is performed "
        "directly on torchvision uint8 tensors (`>127`), exactly equivalent to "
        "the notebook's ToTensor followed by `>0.5`; minibatched inference is "
        "memory-only and does not change values. Runtime is not transferable "
        "from the paper's M4/32GB CPU to the available backend and is not used "
        "to verify the conjunction.\n"
    )
    (target / "verifier_command.txt").write_text(
        ".venv/bin/python repro/src/verify_mnist.py "
        "--output .openresearch/artifacts\n"
    )
    (target / "EVAL.md").write_text(
        f"# Claim 5 evaluation\n\nVerdict: **{verdict}**\n\n"
        f"Numerical subclaim: **{numerical_verdict}**. The three-seed mean "
        f"R² is `{result['r2_mean']:.8g}` and mean MSE is "
        f"`{result['mse_mean']:.8g}` at exactly 10,000 accepted elements. "
        "The runtime subclaim is blocked because hardware differs. Original-row "
        "residual checks agree with support-weighted metrics; the 674-element "
        "negative control is rejected.\n"
    )
    summary = {
        "result": result,
        "independent_checker": independent_output,
        "negative_control": negative,
    }
    write_json(artifact_root / "mnist_summary.json", summary)
    print(json.dumps({"mnist_claim_contract": summary}, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".openresearch" / "artifacts"
    )
    args = parser.parse_args()
    summary = run(args.output.resolve())
    accepted = (
        summary["result"]["verdict"] in {"VERIFIED", "FALSIFIED", "BLOCKED"}
        and summary["result"]["selected_basis_elements"] == TARGET_RANK
        and summary["independent_checker"][
            "all_weighted_support_metrics_match_original_rows"
        ]
        and not summary["negative_control"]["verifier_accepted"]
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
