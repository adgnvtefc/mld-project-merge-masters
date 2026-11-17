"""
Paired bootstrap over dev windows, for the report's confidence intervals on loss differences.

The score is the mean over domains of each domain's mean window loss (all windows are 2,048 tokens), so
resampling is stratified: each domain's windows are resampled on their own, with the same indices for
both runs.

    diff = paired_bootstrap(np.load(a), np.load(b))   # npz files from results/<run_id>/window_losses.npz
    diff["mean"]["delta"], diff["mean"]["lo"], diff["mean"]["hi"]   # a - b; negative means a is better
"""

import numpy as np

from lib.common import DOMAINS


def paired_bootstrap(a, b, n=10_000, level=0.95, seed=0):
    """{domain or 'mean': {'delta', 'lo', 'hi'}} for run a minus run b."""
    rng = np.random.default_rng(seed)
    tail = (1 - level) / 2 * 100
    per_domain, out = [], {}
    for d in DOMAINS:
        diff = np.asarray(a[d], dtype=np.float64) - np.asarray(b[d], dtype=np.float64)
        if diff.shape[0] == 0 or np.asarray(a[d]).shape != np.asarray(b[d]).shape:
            raise ValueError(f"{d}: the two runs were scored on different windows")
        samples = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
        per_domain.append(samples)
        out[d] = {"delta": float(diff.mean()), "lo": float(np.percentile(samples, tail)),
                  "hi": float(np.percentile(samples, 100 - tail))}
    overall = np.mean(per_domain, axis=0)
    out["mean"] = {"delta": float(np.mean([out[d]["delta"] for d in DOMAINS])),
                   "lo": float(np.percentile(overall, tail)), "hi": float(np.percentile(overall, 100 - tail))}
    return out
