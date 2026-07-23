import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


@pytest.fixture(scope="session")
def mushroom_claim():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro" / "src" / "verify_mushrooms.py"),
            "--output",
            str(ARTIFACTS),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((ARTIFACTS / "mushroom_summary.json").read_text())


def test_claim_4_faithful_full_scale_protocol(mushroom_claim):
    result = mushroom_claim["result"]
    assert result["verdict"] == "FALSIFIED"
    assert result["rows"] == 8124
    assert result["variables"] == 22
    assert result["hypergrid_size"] > 10**13
    assert result["authors_default_protocol"]["selected_basis_rank"] == 86


def test_claim_4_mse_directly_contradicts_reported_scale(mushroom_claim):
    result = mushroom_claim["result"]
    assert result["authors_default_protocol"]["independent_mse"] > 1e-12
    assert result["mse_ratio_observed_to_claimed"] > 1000
    assert not result["timing_used_for_verdict"]


def test_claim_4_independent_checker_and_negative_control(mushroom_claim):
    assert mushroom_claim["independent_checker"][
        "all_reported_metrics_match_raw_recomputation"
    ]
    assert not mushroom_claim["negative_control"]["verifier_accepted"]
