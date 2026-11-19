# 20261119-1535_analysis_swaps_hz

Hypothesis: Re-run with the fixed pool-val windows: which part of an expert (embeddings, LayerNorms, attention, MLP) carries its gain, now with valid math numbers.

Status: done

Re-run of 20261118-2127_analysis_swaps_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first n in a row); same command and settings. All four domains' windows changed, so every column differs a little from the original, and math is now valid.

## Outcome

Same conclusions as the original, now holding for math too: swapping embeddings either way is catastrophic, LayerNorms do not matter, and attention and MLP only work with the model's own embeddings.

## Observations

- Own-domain pool-val mean: generalist 3.131, expert 2.753. Math: generalist 3.151, expert 2.716 (gap 0.435, vs 0.04 on the biased windows).
- Generalist + expert embeddings 4.542; expert with the generalist's embeddings 4.347 (code 4.931 and 4.601). Both far worse than either model in every domain.
- LayerNorm swaps change the loss by at most 0.014.
- Generalist + expert attention 3.463 and + expert MLP 3.443: worse than the generalist in every domain.
- The expert with the generalist's attention keeps 38-46% of its gain (math 45%, code 46%, medicine 38%, history 46%); with the generalist's MLP it keeps none (59-92% worse than the generalist).
