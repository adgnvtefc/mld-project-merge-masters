# 20261119-0522_merging_tsv_hz

Hypothesis: Re-run of 20261118-2112_merging_tsv_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: Orthogonalizing the experts' kept directions (TSV-M) reduces interference and beats the plain low-rank merge.

Status: done | dev mean 3.1599

## Outcome

Same choice and same dev score as 20261118-2112_merging_tsv_hz: the pool-val fix changed nothing that was decided. Dev 3.1599 (original 3.1599). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: {'r': 16, 'lam': 0.1, 'others': 0.0} -> math 3.0785 | code 3.1217 | medicine 2.8115 | history 3.5483 | mean 3.1400
- Original, on the biased windows: best on pool-val: {'r': 16, 'lam': 0.1, 'others': 0.0} -> math 3.1821 | code 3.2316 | medicine 2.8152 | history 3.5448 | mean 3.1934
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
