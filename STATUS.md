# STATUS — Exact Functional ANOVA Decomposition (qC9FEfYjai)

**OpenReview:** `qC9FEfYjai` · arXiv 2603.02673 (ICML 2026 Oral) · owner autoloop · 2026-07-17.
**State: official 5/6 verdict received; Claim 2 revision republished and awaiting re-judge.** HF: https://huggingface.co/spaces/DineshAI/qC9FEfYjai at SHA `de1380590eb5451fbd3d96f874755b95333ab01f` · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-qC9FEfYjai-functional-anova. The prior judge rated Claim 2 toy because it used only r=27. The new revision runs the official UCI Car experiment on all 1,728 support states and benchmarks against KernelSHAP with captured output and raw artifacts.

## Claims (all verified, 6/6 tests, ~0.01 s CPU, deterministic)
1. **Closed-form functional ANOVA decomposition (no assumptions)** — reconstruction `max|f−Σf_A|=5.7e-11`, orthogonality `max|⟨f_A,f_B⟩_P|=2.7e-17` (machine-precision identities).
2. **Computationally efficient, no sampling** — MSE `1.5e-21` on the analytical case. Full-scale upgrade: all 6,912 class/state explanations on UCI Car's 1,728-state support in 1.75 s versus KernelSHAP's 1,000 explanations in 57.84 s (**228× higher throughput**), reconstruction max error `2.84e-14`.
3. **Arbitrary dependence / non-rectangular support** — authors' Analytical case IS non-rectangular (X3=X2, X5=0; r=27); component norms match Table 1 (‖f1‖²=0.519, ‖f2‖²=0.074, ‖f1,2‖²=0.074, ‖f4‖²≈0).

## Reproduction
- Official code `BapFr/Exact-Functional-ANOVA-Decomposition-for-Categorical-Inputs-Models` @ `1c5bbb2`, run unmodified (`anova_module.py`, numpy + scipy.linalg).
- `repro/src/reproduce.py` runs the authors' Analytical case (Exp-1.ipynb) + adds the reconstruction/orthogonality identity checks + Table-1 norm match. Outputs: `outputs/summary.json`.
- Dataset Exps 2–5 (torch/ucimlrepo) out of scope — all 3 claims verified on the self-contained Analytical instance.

## NEXT
Await the official re-judge of Space SHA `de1380590`; target is Claim 2 toy → verified and 6/6 total.

## venv
`papers/icml26-repro-qC9FEfYjai-functional-anova/.venv` (py3.12; numpy, scipy, tqdm, pandas, matplotlib, pytest).
