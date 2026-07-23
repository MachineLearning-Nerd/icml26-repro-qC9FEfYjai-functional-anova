import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"
sys.path.insert(0, str(ROOT / "repro" / "src"))

from verify_mnist import original_row_metrics  # noqa: E402


def test_original_row_checker_accumulates_float32_responses_in_float64():
    patterns = np.asarray([[0], [1]], dtype=np.int8)
    x_encoded = np.repeat(patterns, [59_999, 1], axis=0)
    responses = np.asarray([[0.12345679], [0.9876543]], dtype=np.float32)
    reconstruction = responses.astype(np.float64) + np.asarray([[0.01], [-0.02]])

    observed = original_row_metrics(
        x_encoded, patterns, responses, reconstruction
    )
    expanded_response = responses.astype(np.float64)[
        np.repeat([0, 1], [59_999, 1])
    ]
    expanded_reconstruction = reconstruction[
        np.repeat([0, 1], [59_999, 1])
    ]
    residual = expanded_reconstruction - expanded_response
    expected_mse = float(np.mean(residual**2))
    expected_variance = float(
        np.mean((expanded_response - expanded_response.mean()) ** 2)
    )

    assert observed["mse"][0] == pytest.approx(expected_mse, abs=1e-15)
    assert observed["r2"][0] == pytest.approx(
        1.0 - expected_mse / expected_variance, abs=1e-12
    )


@pytest.fixture(scope="session")
def mnist_claim():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro" / "src" / "verify_mnist.py"),
            "--output",
            str(ARTIFACTS),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((ARTIFACTS / "mnist_summary.json").read_text())


def test_claim_5_exact_high_setting(mnist_claim):
    result = mnist_claim["result"]
    assert result["verdict"] in {"VERIFIED", "FALSIFIED", "BLOCKED"}
    assert result["training_rows"] == 60_000
    assert result["dimensions"] == 784
    assert result["selected_basis_elements"] == 10_000
    assert len(result["seed_runs"]) == 3


def test_claim_5_independent_original_row_checker(mnist_claim):
    checker = mnist_claim["independent_checker"]
    assert checker["all_weighted_support_metrics_match_original_rows"]
    assert checker["maximum_mse_abs_difference"] < 1e-12
    assert checker["maximum_r2_abs_difference"] < 1e-10


def test_claim_5_negative_control_and_hardware_limitation(mnist_claim):
    assert not mnist_claim["negative_control"]["verifier_accepted"]
    assert mnist_claim["negative_control"]["selected_basis_elements"] == 674
    assert not mnist_claim["result"]["timing_used_for_verdict"]
