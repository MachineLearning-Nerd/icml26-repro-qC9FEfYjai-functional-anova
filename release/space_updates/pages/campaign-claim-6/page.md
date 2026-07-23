# Claim 6 — exact standard SHAP recovery

Verdict: **VERIFIED**

The Car inputs cover their Cartesian product exactly once, so their empirical
distribution is exactly product-uniform. For all 1,728 states, four outputs,
and all 64 coalitions, Equation (5) is compared with exhaustive standard SHAP:

- maximum Equation (5) difference: `8.4377e-15`
- independent checker: all 720 feature permutations
- permutation/coalition difference: `5.0515e-15`
- negative control (`|A|+1` divisor) error: `0.21963` (rejected)

No KernelSHAP approximation is used in this claim test.
