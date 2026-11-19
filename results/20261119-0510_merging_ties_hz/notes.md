# 20261119-0510_merging_ties_hz

Hypothesis: Re-run of 20261118-2012_merging_ties_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: TIES (top-20% trim, sign election) over the five task vectors beats plain task arithmetic.

Status: done | dev mean 3.7196

## Outcome

Same choice and same dev score as 20261118-2012_merging_ties_hz: the pool-val fix changed nothing that was decided. Dev 3.7196 (original 3.7196). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: {'lam': 0.4} -> math 3.5726 | code 3.6702 | medicine 3.4593 | history 4.1143 | mean 3.7041
- Original, on the biased windows: best on pool-val: {'lam': 0.4} -> math 3.6421 | code 3.7795 | medicine 3.4607 | history 4.1065 | mean 3.7472
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
