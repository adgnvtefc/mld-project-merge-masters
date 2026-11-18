# 20261118-2131_analysis_paths_hz

Hypothesis: Whether the straight line between the generalist and each expert (and between experts) has a loss barrier, i.e. whether they sit in one valley.

Status: done

## Outcome

No loss barrier between the generalist and any expert, or between experts: the models sit in one connected valley.

## Observations

- Along generalist -> expert, the expert's own-domain loss falls steadily (code 3.307 -> 2.838, medicine 2.806 -> 2.602, history 3.513 -> 3.065), while the 4-domain mean rises steadily (3.209 -> 3.40-3.67).
- Largest bump above the straight line on the mean: 0.039 (generalist -> code); 0.008 (math -> code); 0.002 (medicine -> history).
- At a = 0.1 toward any expert the mean is already worse than the generalist (3.209 -> 3.209-3.225).
- Run before commit d725d3f, so its pool-val windows were the first n in a row. For math those were 79% one 104K-token GMAT reading-comprehension guide, where the math expert is barely better than the generalist. Math columns here describe that book, not math; code, medicine and history are unaffected.
