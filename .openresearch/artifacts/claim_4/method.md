# Method, limitations, and deviations

The primary row omits XGBoost random_state exactly like the released notebook. Four explicit seed controls estimate implementation uncertainty. The weighted residual is independently recomputed. XGBoost and dependencies are pinned. Timing is descriptive only: the local CPU is not the paper's stated M4/32GB machine. The numerical MSE, which differs by many orders of magnitude, determines the verdict.
