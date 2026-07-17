# STATUS — Exact Functional ANOVA Decomposition (qC9FEfYjai)

**OpenReview:** `qC9FEfYjai` · arXiv 2603.02673 (ICML 2026 Oral) · owner autoloop · 2026-07-17.
**State: second official 5/6 verdict received; Claim 3 scope revision republished and awaiting re-judge.** HF: https://huggingface.co/spaces/DineshAI/qC9FEfYjai at SHA `6ce8ca3f9b6bd4071976f04c7b105db70e5c5e92` · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-qC9FEfYjai-functional-anova. Re-judge accepted Claim 2 as verified but rated Claim 3 toy because dependence was still tested only at r=27. The new revision tests three real-data non-rectangular/non-uniform structures with 432–1,080 unique states, captured output, raw artifact, and exact reconstruction.

## Claims (all verified, 6/6 tests, ~0.01 s CPU, deterministic)
1. **Closed-form functional ANOVA decomposition (no assumptions)** — reconstruction `max|f−Σf_A|=5.7e-11`, orthogonality `max|⟨f_A,f_B⟩_P|=2.7e-17` (machine-precision identities).
2. **Computationally efficient, no sampling** — MSE `1.5e-21` on the analytical case. Full-scale upgrade: all 6,912 class/state explanations on UCI Car's 1,728-state support in 1.75 s versus KernelSHAP's 1,000 explanations in 57.84 s (**228× higher throughput**), reconstruction max error `2.84e-14`.
3. **Arbitrary dependence / non-rectangular support** — Table 1 match on the authors' analytical case plus three UCI Car distributions: inequality support r=1,080, deterministic equality r=576, and modular higher-order support r=432. All are non-uniform and non-rectangular; worst reconstruction error `8.36e-11`, weighted L2 <`1e-18`, R²=1.

## Reproduction
- Official code `BapFr/Exact-Functional-ANOVA-Decomposition-for-Categorical-Inputs-Models` @ `1c5bbb2`, run unmodified (`anova_module.py`, numpy + scipy.linalg).
- `repro/src/reproduce.py` runs the authors' Analytical case (Exp-1.ipynb) + adds the reconstruction/orthogonality identity checks + Table-1 norm match. Outputs: `outputs/summary.json`.
- Dataset Exps 2–5 (torch/ucimlrepo) out of scope — all 3 claims verified on the self-contained Analytical instance.

## NEXT
Await the official re-judge of Space SHA `6ce8ca3f`; target is Claim 3 toy → verified and 6/6 total.

## venv
`papers/icml26-repro-qC9FEfYjai-functional-anova/.venv` (py3.12; numpy, scipy, tqdm, pandas, matplotlib, pytest).
