# Claim 3 — Algorithm 1 complexity

Verdict against the live judge wording: **FALSIFIED**

The paper HTML does not itself explicitly claim an end-to-end `O(r³)` bound;
this page tests the stronger wording in the live verdict record. For the valid
one-variable support family `{M-3, M-2, M-1}`, support cardinality stays
`r=3`, but the official greedy loop evaluates exactly `M-2` dictionary
elements before reaching rank 3.

| Hypergrid M | r | official evaluations | independent evaluations |
|---:|---:|---:|---:|
| 128 | 3 | 126 | 126 |
| 512 | 3 | 510 | 510 |
| 2,048 | 3 | 2,046 | 2,046 |
| 8,192 | 3 | 8,190 | 8,190 |

A nearby support `{0,1,M-1}` stops after two evaluations, and a tampered count
is rejected. The result falsifies ambient-size-independent end-to-end `O(r³)`;
it does not attribute wording to the paper that is absent from the source.
