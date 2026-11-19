# 20261119-1537_analysis_tokens_hz

Hypothesis: Re-run with the fixed pool-val windows: whether each expert's gain is spread over all tokens or concentrated in rare ones, now including math.

Status: done

Re-run of 20261118-2130_analysis_tokens_hz after the pool-val fix (d725d3f: windows spread over the whole slice, not the first n in a row); same command and settings. All four domains' windows changed, so every column differs a little from the original, and math is now valid.

## Outcome

Spread over most tokens and token types, larger per occurrence on rare tokens, now also for math. One change from the original: the gain is largest in a window's first 16 tokens and falls somewhat after, rather than being flat; most of it remains to the end.

## Observations

- Expert better on 62-65% of tokens; the top 100 token types hold 17-25% of the positive gain; the generalist is better on 26-30% of token types.
- Mean gain by frequency quintile, rarest to commonest: math 0.78 -> 0.13, code 0.79 -> 0.11, medicine 0.45 -> 0.05, history 0.85 -> 0.09.
- Mean gain by position (0-16 / 16-128 / 128-512 / 512-2048): math 0.62 / 0.51 / 0.48 / 0.42, code 0.88 / 0.48 / 0.46 / 0.39, medicine 0.29 / 0.25 / 0.23 / 0.23, history 0.54 / 0.48 / 0.39 / 0.41. The 0-16 bucket is only 256 tokens per domain.
- Top math types by total gain: ' of', newline, '.', ',', ' and', '%%', ' ###'; the others are also led by common tokens.
- The original's flat position profile (code 0.49 / 0.49 / 0.53 / 0.44) came from 16 consecutive windows; with windows spread over many documents the early-position gain is larger.
