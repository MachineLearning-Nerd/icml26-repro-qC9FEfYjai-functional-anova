import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"


def s():
    return json.loads((OUT / "summary.json").read_text())


def b():
    return json.loads((OUT / "car_efficiency_summary.json").read_text())


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


def test_c2_real_dataset_is_not_toy_scale():
    assert b()["claim_2_full_scale"] == "verified"
    assert b()["rows"] == 1728
    assert b()["full_support_size"] == 1728
    assert b()["kernel_output_explanations"] >= 1000


def test_c2_exact_method_is_exact_and_faster_than_sampling_baseline():
    assert b()["max_reconstruction_abs_error"] < 1e-10
    assert b()["throughput_ratio_exact_over_kernel"] > 10
