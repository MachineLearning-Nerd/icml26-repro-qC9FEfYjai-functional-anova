# Method, limitations, and deviations

Use the 60,000-image binarized MNIST training split, all 784 pixels, the
published `BinaryBasisExtractor`, 10,000 selected elements, spatial-neighbor
pairs, and deterministic MLP seeds 0, 1, and 2. An independent checker expands
support predictions back to all 60,000 rows. A 674-basis solve is the negative
control.

The paper does not pin the training seed or hardware. Therefore the numerical
result near the rounding boundary and the timing subclaim remain **BLOCKED**.
