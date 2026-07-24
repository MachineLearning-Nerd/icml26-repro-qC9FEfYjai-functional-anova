# Claim 1 — full-scale hierarchical orthogonality

Verdict: **VERIFIED**

The earlier 27-point example remains preserved, but is not used for full-scale
credit. On the complete UCI Car domain (1,728 states, six categorical
variables, four class outputs), the verifier checks every nonconstant
component, every proper lower subset, and every conditioning configuration:
2,660 conditions in total.

- maximum absolute conditional mean: `3.7354e-16`
- independent indicator-moment maximum: `9.3386e-17`
- threshold: `1e-10`
- negative control: a `1e-3` lower-order contamination is rejected

Data SHA-256:
`b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83`.
