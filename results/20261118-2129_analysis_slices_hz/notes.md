# 20261118-2129_analysis_slices_hz

Hypothesis: Checks whether the pool files are ordered, so that pool-val (the last 2M tokens) differs from the rest; the math expert barely beats the generalist on pool-val but leads by 0.47 on dev.

Status: done

## Outcome

The pool files are not ordered: each expert's own-domain lead is roughly constant along its file, including the held-out tail.

## Observations

- Gap (generalist - expert) at 10 places: math 0.40-0.55, code 0.41-0.52, medicine 0.17-0.20, history 0.23-0.49.
- This pointed to the real cause of the odd math pool-val numbers: one long document at the start of pool-val (see commit d725d3f).
