# Claim 4 — UCI Mushrooms numerical result

Verdict for the numerical MSE claim: **FALSIFIED**

The faithful UCI Mushroom protocol uses 8,124 rows, 22 categorical variables,
and hypergrid size `121,899,810,816,000`. Across the authors' default and seeds
0, 1, 2, and 42:

- R²: `0.9999920823`
- independently recomputed MSE: `1.9673189361e-6`
- claimed MSE: approximately `1e-15`
- observed/claimed ratio: `1.9673e9`
- maximum reconstruction error: `0.0229444`

The raw residual checker agrees exactly in MSE and within `7.41e-14` in R².
A tampered `1e-15` MSE is rejected. Timing is reported but excluded from the
verdict because the paper's hardware is not reproduced.
