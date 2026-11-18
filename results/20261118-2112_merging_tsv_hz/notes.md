# 20261118-2112_merging_tsv_hz

Hypothesis: Orthogonalizing the experts' kept directions (TSV-M) reduces interference and beats the plain low-rank merge.

Status: done | dev mean 3.1599

## Outcome

Dev 3.1599, worse than the generalist by 0.0044 (95% CI [+0.0042, +0.0045]) and worse on every domain, but better than the plain low-rank merge (3.1700). Supports TSV over plain low-rank; does not beat the generalist.

## Observations

- Best pool-val point: r 16, lam 0.1, others kept at the generalist -> 3.1934 (generalist 3.1888). Again the smallest r and lam in the grid; lam = 0 was not in the grid.
- At every matching (r, lam) the orthogonalized merge hurt less than the plain one, e.g. r 128, lam 0.25: 3.28 vs 3.36-3.40; lam 0.5: 3.60 vs 4.05-4.25.
- Per domain vs the generalist: math +0.0028, code +0.0062, medicine +0.0044, history +0.0042. No extra parameters. 213 s, 0.059 A10G-hours.

## Caveat (added 2026-11-18)

Settings and pool-val numbers in this run used the first n pool-val windows in a row. For math these were 79% one GMAT reading-comprehension guide, where the math expert is barely better than the generalist, so math pulled every choice toward the generalist. Dev scores are correct. Fixed in commit d725d3f; this run has not been re-run. See error_analysis.md.
