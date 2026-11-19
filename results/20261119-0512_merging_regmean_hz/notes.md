# 20261119-0512_merging_regmean_hz

Hypothesis: Re-run of 20261118-2014_merging_regmean_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: RegMean is the best closed-form merge and beats the generalist (3.156).

Status: done | dev mean 3.4446

## Outcome

Same choice and same dev score as 20261118-2014_merging_regmean_hz: the pool-val fix changed nothing that was decided. Dev 3.4446 (original 3.4446). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: alpha 0.5 -> math 3.2372 | code 3.3484 | medicine 3.2956 | history 3.8444 | mean 3.4314
- Original, on the biased windows: best on pool-val: alpha 0.5 -> math 3.3328 | code 3.4543 | medicine 3.2824 | history 3.8526 | mean 3.4805
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
