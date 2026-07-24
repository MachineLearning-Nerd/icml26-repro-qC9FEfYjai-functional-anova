# Claim 2 — full-support Gamma system

Verdict: **VERIFIED**

For all four UCI Car class-probability functions, the reproduction constructs
the 1,728×1,728 full-support basis and directly solves
`Gamma c(f) = mu(f)`.

- maximum `|Gamma c - mu|`: `3.5527e-15`
- maximum reconstruction error over 6,912 class/state values: `6.9719e-15`
- independent direct `B c = f` reconstruction error: `2.7520e-14`
- maximum coefficient difference between the two solves: `1.2212e-15`
- negative control reconstruction error: `4.0e-3` (rejected)

This directly tests Equation (16), rather than using throughput as a proxy.
