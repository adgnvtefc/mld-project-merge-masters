# 20261118-2130_analysis_tokens_hz

Hypothesis: Whether each expert's gain is spread over all tokens or concentrated in rare domain tokens, and where in the window it appears.

Status: done

## Outcome

Each expert's gain is spread over most tokens and token types, larger per occurrence on rare tokens, and present at every position in the window.

## Observations

- Expert better on 57-65% of tokens; top 100 token types hold only 19-27% of the positive gain; the generalist is better on 24-37% of token types.
- Mean gain by frequency quintile, rarest to commonest: code 0.97 -> 0.11, history 0.98 -> 0.09, medicine 0.33 -> 0.03.
- Gain by position is roughly flat (for example code 0.49 / 0.49 / 0.53 / 0.44 for positions 0-16 / 16-128 / 128-512 / 512-2048), so it is not just a prior that context quickly overrides.
- Top types by total gain are common ones ('.', ' the', newline, ','); decoded lists are in error_analysis.md.
- Run before commit d725d3f, so its pool-val windows were the first n in a row. For math those were 79% one 104K-token GMAT reading-comprehension guide, where the math expert is barely better than the generalist. Math columns here describe that book, not math; code, medicine and history are unaffected.
