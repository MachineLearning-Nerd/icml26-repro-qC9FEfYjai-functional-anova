# Claim 5 — 784-dimensional, 10,000-basis MNIST

Verdict: **BLOCKED**

This is a completed high-dimensional experiment, not a skip or proxy:
60,000 binarized MNIST training images, 784 pixels, 59,984 unique support
points, the published basis extractor, exactly 10,000 selected elements, and
deterministic seeds 0, 1, and 2.

| Seed | R² | MSE |
|---:|---:|---:|
| 0 | 0.8653349373 | 0.0115653374 |
| 1 | 0.8661102020 | 0.0116552603 |
| 2 | 0.8650972249 | 0.0120567052 |

Mean R² is `0.8655141214` (SD `0.0005297271`). The independent checker expands
back to all 60,000 rows and agrees to `2.00e-15` in R² and `8.33e-17` in MSE.
A 674-basis negative control falls to R² `0.538–0.543` and is rejected.

The paper omits the model seed and comparable hardware. All results are very
near its rounded `0.86`, but just outside the predeclared verification interval;
they are not far enough away for a robust falsification. The 15-minute
subclaim is also not judged on different hardware.
