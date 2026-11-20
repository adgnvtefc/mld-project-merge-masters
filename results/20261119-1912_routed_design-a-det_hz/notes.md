# 20261119-1912_routed_design-a-det_hz

Hypothesis: With deterministic calibration, two builds of design A give bit-identical weights (same fingerprint), the same routing and the same dev score; build 1 of 2.

Status: done

Parent: 20261119-1850_routed_design-a-commit_hz

## Outcome

Supported for determinism (see its twin 20261119-1921: identical weights fingerprint 642d1f1c576e6348, identical pool-val numbers and routing, identical dev 3.0448 down to every window). But dev is 0.0035 worse than 20261119-1850 (3.0413; 95% range +0.0018 to +0.0051).

## Observations

- Cause of the 0.0035: calibration ran at batch 1 (needed for the deterministic attention path), and the taylor statistic took |sum h*g| per batch, so it became a per-window statistic and pruned a different set of neurons (per-layer widths moved by up to about 150). Fixed in 60d1b37: the statistic now sums over fixed groups of 4 windows. Re-run as 20261119-1936.
- Pool-val now uses 256 windows per domain, so its numbers are not comparable with the 64-window runs (oracle 3.03-3.04 here vs 2.999); dev is unaffected.
- Commit point chosen on pool-val: 224 (pool-val falls steadily from 128 to 224 on 256 windows, which the 64-window runs could not show).
- 0.10 A10G-h including dev scoring.
