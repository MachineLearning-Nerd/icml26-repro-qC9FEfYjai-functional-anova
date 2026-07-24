# Publication status — awaiting judge

Paper: *Exact Functional ANOVA Decomposition for Categorical Inputs Models*
(arXiv `2603.02673`, OpenReview `qC9FEfYjai`).

- Live judged score at publication: **3/12**
- Previous judged Space revision:
  `6ce8ca3f9b6bd4071976f04c7b105db70e5c5e92`
- Published candidate revision:
  `921044bf0219e121c4574f72fdfd65c500e95658`
- State: **AWAITING JUDGE REEVALUATION**
- Scientific evidence commit:
  `4b3a0670f7caa28c84599e884ceeb9d77981f23d`
- Cumulative run:
  `ad3897e1-589b-4eac-b902-b81b60c41a31`

The cumulative run printed `27 passed in 16141.26s (4:29:01)` before its
Hugging Face job timeout. The publication added exactly eight allowlisted text
files, performed no deletions, preserved every protected path, and verified all
published hashes. No judge-score increase is claimed before reevaluation.

## Historical repository status preserved from 2026-07-17

The following text is retained verbatim as the repository's pre-campaign status
record. It is historical evidence and is not the current live-judge state.

> # STATUS — Exact Functional ANOVA Decomposition (qC9FEfYjai)
>
> **OpenReview:** `qC9FEfYjai` · arXiv 2603.02673 (ICML 2026 Oral) · owner autoloop · 2026-07-17.
> **State: official verdict complete; high quality; 6/6 points.** HF: https://huggingface.co/spaces/DineshAI/qC9FEfYjai at SHA `6ce8ca3f9b6bd4071976f04c7b105db70e5c5e92` · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-qC9FEfYjai-functional-anova. All three claims were officially verified at `2026-07-17T03:41:36+00:00` after adding the full-scale efficiency benchmark and three non-toy dependent-support stress tests.
>
> ## Claims (all verified, 6/6 tests, ~0.01 s CPU, deterministic)
>
> 1. **Closed-form functional ANOVA decomposition (no assumptions)** — reconstruction `max|f−Σf_A|=5.7e-11`, orthogonality `max|⟨f_A,f_B⟩_P|=2.7e-17` (machine-precision identities).
> 2. **Computationally efficient, no sampling** — MSE `1.5e-21` on the analytical case. Full-scale upgrade: all 6,912 class/state explanations on UCI Car's 1,728-state support in 1.75 s versus KernelSHAP's 1,000 explanations in 57.84 s (**228× higher throughput**), reconstruction max error `2.84e-14`.
> 3. **Arbitrary dependence / non-rectangular support** — Table 1 match on the authors' analytical case plus three UCI Car distributions: inequality support r=1,080, deterministic equality r=576, and modular higher-order support r=432. All are non-uniform and non-rectangular; worst reconstruction error `8.36e-11`, weighted L2 <`1e-18`, R²=1.
>
> ## Reproduction
>
> - Official code `BapFr/Exact-Functional-ANOVA-Decomposition-for-Categorical-Inputs-Models` @ `1c5bbb2`, run unmodified (`anova_module.py`, numpy + scipy.linalg).
> - `repro/src/reproduce.py` runs the authors' Analytical case (Exp-1.ipynb) + adds the reconstruction/orthogonality identity checks + Table-1 norm match. Outputs: `outputs/summary.json`.
> - Dataset Exps 2–5 (torch/ucimlrepo) out of scope — all 3 claims verified on the self-contained Analytical instance.
>
> ## OFFICIAL RESULT
>
> High quality; C1 verified, C2 verified, C3 verified; **6/6 points**.
>
> ## venv
>
> `papers/icml26-repro-qC9FEfYjai-functional-anova/.venv` (py3.12; numpy, scipy, tqdm, pandas, matplotlib, pytest).
