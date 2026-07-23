import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


@pytest.fixture(scope="session")
def exact_claims():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "repro" / "src" / "verify_exact_claims.py"),
            "--output",
            str(ARTIFACTS),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((ARTIFACTS / "exact_claims_summary.json").read_text())


def test_claim_1_full_scale_hierarchical_orthogonality(exact_claims):
    result = exact_claims["1"]["result"]
    assert result["verdict"] == "VERIFIED"
    assert result["states"] == 1728
    assert result["hierarchical_conditions_checked"] > 500
    assert result["max_abs_conditional_mean"] < 1e-10


def test_claim_1_independent_checker_and_negative_control(exact_claims):
    assert exact_claims["1"]["independent_checker"]["accepted"]
    assert not exact_claims["1"]["negative_control"]["verifier_accepted"]


def test_claim_2_full_support_gamma_linear_system(exact_claims):
    result = exact_claims["2"]["result"]
    assert result["verdict"] == "VERIFIED"
    assert result["basis_shape"] == [1728, 1728]
    assert result["systems_solved"] == 4
    assert result["max_abs_gamma_c_minus_mu"] < 1e-10
    assert result["max_abs_reconstruction_error"] < 1e-10


def test_claim_2_independent_checker_and_negative_control(exact_claims):
    assert exact_claims["2"]["independent_checker"]["accepted"]
    assert not exact_claims["2"]["negative_control"]["verifier_accepted"]


def test_claim_3_fixed_r_counterexample(exact_claims):
    result = exact_claims["3"]["result"]
    assert result["verdict"] == "FALSIFIED"
    assert {row["support_cardinality_r"] for row in result["rows"]} == {3}
    assert result["rows"][-1]["candidate_evaluations"] == 8190


def test_claim_3_independent_checker_and_negative_control(exact_claims):
    assert exact_claims["3"]["independent_checker"][
        "all_counts_match_official_instrumentation"
    ]
    assert not exact_claims["3"]["negative_control"][
        "verifier_accepted_tampered_result"
    ]


def test_claim_6_exact_standard_shap_recovery(exact_claims):
    result = exact_claims["6"]["result"]
    assert result["verdict"] == "VERIFIED"
    assert result["states_checked"] == 1728
    assert result["empirical_distribution_is_exact_product_uniform"]
    assert result["max_abs_equation5_vs_exact_standard_shap"] < 1e-10


def test_claim_6_independent_checker_and_negative_control(exact_claims):
    assert exact_claims["6"]["independent_checker"]["accepted"]
    assert not exact_claims["6"]["negative_control"]["verifier_accepted"]
