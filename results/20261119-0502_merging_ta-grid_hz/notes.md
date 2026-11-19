# 20261119-0502_merging_ta-grid_hz

Hypothesis: Re-run of 20261118-2004_merging_ta-grid_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: Seed-relative task arithmetic including the generalist's task vector, with coefficients picked on pool-val, beats the generalist (3.156).

Status: done | dev mean 3.1555

## Outcome

Same choice and same dev score as 20261118-2004_merging_ta-grid_hz: the pool-val fix changed nothing that was decided. Dev 3.1555 (original 3.1555). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: {'lam_e': 0.0, 'lam_g': 1.0} -> math 3.0769 | code 3.1156 | medicine 2.8073 | history 3.5445 | mean 3.1361
- Original, on the biased windows: best on pool-val: {'lam_e': 0.0, 'lam_g': 1.0} -> math 3.1801 | code 3.2242 | medicine 2.8105 | history 3.5401 | mean 3.1887
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
