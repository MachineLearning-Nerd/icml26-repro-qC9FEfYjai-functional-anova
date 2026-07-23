# Method, limitations, and deviations

On the real 1,728-state UCI Car full hypergrid, form the published basis `B`,
`Gamma = B.T diag(P) B`, and `mu = B.T diag(P) f`. Solve all four class-output
systems. Independently solve `B c = f` without forming Gamma. The negative
control perturbs one nonconstant coefficient by `1e-3`.

Limitation: this is one full-scale real categorical domain, with exact
product-uniform full support.
