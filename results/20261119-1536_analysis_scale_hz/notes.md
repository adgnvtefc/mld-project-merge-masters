# 20261119-1536_analysis_scale_hz

Hypothesis: Re-run with the fixed pool-val windows: whether failed merges are miscalibrated or have shrunken hidden states.

Status: done

Re-run of 20261118-2130_analysis_scale_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first n in a row); same command and settings. All four domains' windows changed, so every column differs a little from the original, and math is now valid.

## Outcome

Same conclusions as the original: the best output temperature is 1.0 for every model except the overshooting mix, where it recovers only 0.23 of 2.22.

## Observations

- Pool-val: generalist 3.130, math expert 3.321, average of experts 3.503, learned-lambda (re-run 20261119-0527) 3.116, ta lam_e 0.1 lam_g 1.0 5.350 (5.121 at T 0.8).
- Hidden-state RMS per block matches the generalist for the math expert, the average and the learned mix; the overshooting mix's is about 40% lower in every block (0.04-0.11 vs 0.06-0.19).
- The --learned path pointed to the re-run learned mix, and the log shows it was loaded.
