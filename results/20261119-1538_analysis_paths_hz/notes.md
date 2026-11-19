# 20261119-1538_analysis_paths_hz

Hypothesis: Re-run with the fixed pool-val windows: whether straight lines between the generalist and experts have a loss barrier.

Status: done

Re-run of 20261118-2131_analysis_paths_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first n in a row); same command and settings. All four domains' windows changed, so every column differs a little from the original, and math is now valid.

## Outcome

Same conclusion as the original: no loss barrier between the generalist and any expert, or between experts.

## Observations

- Along generalist -> expert, the expert's own-domain loss falls steadily (math 3.151 -> 2.716, code 2.985 -> 2.532, medicine 2.729 -> 2.532, history 3.658 -> 3.235), while the 4-domain mean rises (3.131 -> 3.32-3.54).
- Largest bump above the straight line on the mean: 0.037 (generalist -> code), 0.016 (math -> code), 0.001 (generalist -> math), 0.000 for the rest.
- At a = 0.1 toward any expert the mean is already no better than the generalist (3.1305 -> 3.1312-3.1428). Toward math it is nearly flat: math -0.041, medicine +0.029.
