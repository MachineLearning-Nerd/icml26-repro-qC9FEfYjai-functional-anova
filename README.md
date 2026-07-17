# Repro — Exact Functional ANOVA Decomposition (ICML 2026 Oral, qC9FEfYjai)

Reproduction of *Exact Functional ANOVA Decomposition for Categorical Inputs Models* (arXiv 2603.02673) for the ICML 2026 Agent Reproduction Challenge. CPU exact-instance: verify the closed-form ANOVA decomposition by exact identities + machine-precision reference match on the authors' Analytical case (non-rectangular dependent support).

## Claims (all verified, 6/6 tests, ~0.01 s CPU)
1. Closed-form functional ANOVA decomposition (no assumptions) — reconstruction 5.7e-11, orthogonality 2.7e-17.
2. Computationally efficient, no sampling — MSE 1.5e-21 (machine precision), 0.008 s.
3. Arbitrary dependence / non-rectangular support — Table 1 norms match (‖f1‖²=0.519, ‖f2‖²=0.074, ‖f1,2‖²=0.074, ‖f4‖²≈0).

## Reproduce
```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy tqdm pandas matplotlib pytest
git clone --depth 1 https://github.com/BapFr/Exact-Functional-ANOVA-Decomposition-for-Categorical-Inputs-Models upstream
.venv/bin/python repro/src/reproduce.py --output outputs
.venv/bin/python -m pytest repro/test_reproduction.py
```
Logbook: https://huggingface.co/spaces/DineshAI/qC9FEfYjai
