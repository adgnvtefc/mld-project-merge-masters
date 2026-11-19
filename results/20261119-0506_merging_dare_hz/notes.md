# 20261119-0506_merging_dare_hz

Hypothesis: Re-run of 20261118-2008_merging_dare_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first 64 in a row); same grid and settings. Original hypothesis: DARE (drop 90%, rescale) before task arithmetic reduces interference and beats plain task arithmetic.

Status: done | dev mean 3.4684

## Outcome

Same choice and same dev score as 20261118-2008_merging_dare_hz: the pool-val fix changed nothing that was decided. Dev 3.4684 (original 3.4684). Dev per-window losses are identical to the original's.

## Observations

- Chosen on fixed pool-val: best on pool-val: {'lam_e': 0.0, 'lam_g': 0.0} -> math 3.3947 | code 3.4371 | medicine 3.0965 | history 3.8699 | mean 3.4496
- Original, on the biased windows: best on pool-val: {'lam_e': 0.0, 'lam_g': 0.0} -> math 3.4730 | code 3.5547 | medicine 3.1022 | history 3.8614 | mean 3.4979
- Pool-val is now lower in math and code because the windows no longer sit on one GMAT reading guide; the merges are still worse than the generalist in every domain, so the same edge point wins.
