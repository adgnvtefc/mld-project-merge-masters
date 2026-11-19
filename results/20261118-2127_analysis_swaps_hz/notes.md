# 20261118-2127_analysis_swaps_hz

Hypothesis: Which part of an expert (embeddings, LayerNorms, attention, MLP) carries its gain, and which parts only work together.

Status: done

## Outcome

Swapping embeddings either way is catastrophic; LayerNorms do not matter; attention and MLP only work as a pair with the model's own embeddings.

## Observations

- Own-domain pool-val mean: generalist 3.209, expert 2.919. Generalist + expert embeddings 4.821; expert with the generalist's embeddings 4.770.
- LayerNorm swaps change the loss by at most 0.01 in either direction.
- Generalist + expert attention 3.555 and + expert MLP 3.566 (both worse than the generalist). The expert with the generalist's attention keeps 35-52% of its gain (code, medicine, history); with the generalist's MLP it keeps none (3.567).
- Ratios for math are meaningless here (the biased math windows give a gap of 0.04).
- Run before commit d725d3f, so its pool-val windows were the first n in a row. For math those were 79% one 104K-token GMAT reading-comprehension guide, where the math expert is barely better than the generalist. Math columns here describe that book, not math; code, medicine and history are unaffected.

Re-run as 20261119-1535_analysis_swaps_hz after the fix (2026-11-19), with every domain's windows spread over pool-val: see that run for the corrected numbers; error_analysis.md uses them.
