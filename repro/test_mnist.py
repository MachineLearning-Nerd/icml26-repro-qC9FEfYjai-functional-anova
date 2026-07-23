import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


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
