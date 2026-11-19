# 20261119-0516_merging_low-rank_hz

Hypothesis: Re-run of 20261118-2107_merging_low-rank_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: Adding each expert's top-r change from the generalist (no extra parameters) beats the generalist and the learned mix (3.141).

Status: done | dev mean 3.1700

## Outcome

Same choice and same dev score as 20261118-2107_merging_low-rank_hz: the pool-val fix changed nothing that was decided. Dev 3.1700 (original 3.1700). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: {'r': 16, 'lam': 0.1, 'others': 1.0} -> math 3.0852 | code 3.1381 | medicine 2.8226 | history 3.5568 | mean 3.1507
- Original, on the biased windows: best on pool-val: {'r': 16, 'lam': 0.1, 'others': 1.0} -> math 3.1868 | code 3.2490 | medicine 2.8270 | history 3.5534 | mean 3.2041
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
