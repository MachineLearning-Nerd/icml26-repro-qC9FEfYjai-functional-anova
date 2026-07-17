import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"


def s():
    return json.loads((OUT / "summary.json").read_text())


def test_c1_closed_form_decomposition():
    assert s()["claim_1"] == "verified"


def test_c2_efficient_no_sampling():
    assert s()["claim_2"] == "verified"


def test_c3_non_rectangular_support():
    assert s()["claim_3"] == "verified"


def test_reconstruction_machine_precision():
    assert s()["reconstruction_max_abs_err"] < 1e-10


def test_orthogonality():
    assert s()["orthogonality_max_abs_err"] < 1e-10


def test_table1_norms():
    n = s()["norms"]
    assert abs(n["(1,)"] - 0.518) < 2e-3
    assert abs(n["(2,)"] - 0.074) < 2e-3
    assert abs(n["(1, 2)"] - 0.074) < 2e-3
    assert n["(4,)"] < 1e-6
