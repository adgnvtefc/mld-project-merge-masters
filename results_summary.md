# Results summary

> **Newer results:** this page covers our first method, mixing weights. Two later methods route each text to
> one expert. Pruned Expert Stacks scores 3.042 with no training, 2.973 after 30 minutes of
> training, 2.963 after 90 minutes, 2.957 with better
> embeddings, 2.927 with a lower learning rate, and 2.899 with
> distillation from each expert, our best so far. Ayush's trained Shared-Attention MoE scores 2.989. See `analysis.md`, sections 7
> to 9.

Last updated: November 19, 2026. Runs by Hanmo on our EC2 machine.

> **A bug we found and fixed.** Every mixing method picks its settings on a slice of training text we
> set aside. For math, that slice was mostly one long book: a GMAT reading guide, not math. We fixed it
> and re-ran every affected method on November 19. **Every method chose the same settings and got the
> same score**, so the results below stand. We also re-ran the checks behind `error_analysis.md`, which
> explains why mixing failed. Its conclusions held too.

## How to read the numbers

Every model gets one **score**. The score measures how wrong the model is when it guesses the next
word. **Lower is better.** The score is averaged over four kinds of text: math, code, medicine and
history.

Differences look small, but they matter. A drop of 0.01 is a real gain in this project.

## The goal

We have one shared starting model. Four copies of it were trained further, each on one kind of text.
These are the four **experts**. A fifth copy was trained on a general mix. This is the **generalist**.

We want one model that does well on all four kinds of text. Two numbers frame the goal:

- The **generalist scores 3.156.** This is the bar to beat.
- If each kind of text could use its own expert, the average would be **2.78.** This is the best we
  could hope for.

## The provided models

We scored each provided model as is. All of our numbers match the ones the course gave us, so we
trust our scoring setup.

| Model | Score | In plain words |
|---|---|---|
| Generalist | 3.156 | The best single model |
| Math expert | 3.331 | Great at math (2.67), weak elsewhere |
| History expert | 3.439 | Good at history, weak elsewhere |
| Medicine expert | 3.466 | Great at medicine (2.59), weak elsewhere |
| Starting model | 3.468 | The shared starting point of all the others |
| Code expert | 3.527 | Great at code (2.71), weakest elsewhere |
| Plain average of the 4 experts | 3.516 | Worse than every model it was built from except the code expert |
| Course example: generalist trained 6 more minutes | 3.163 | Slightly worse than the generalist |

Each expert is very good at its own kind of text and poor at the rest.

## What we tried: mixing the models without training them

Our first method takes the numbers inside the models and blends them into one model. No model is
trained on new text.

Each method below has a setting that controls how much of each model goes into the mix. To pick that
setting, we used a slice of our training text that we set aside. We never used the course's scoring text
for this. The course rules forbid it.

| Method | Score | Better than the generalist? |
|---|---|---|
| **Learned mix** | **3.141** | **Yes, by 0.014** |
| Simple mix | 3.156 | No. Its best choice was the generalist itself. |
| Best-fit mix | 3.445 | No |
| Drop-and-boost mix | 3.468 | No. Its best choice was the starting model itself. |
| Keep-the-biggest-changes mix | 3.720 | No. Worse than the starting model. |

What each method does:

- **Simple mix.** Add some of each expert and some of the generalist to the starting model. We tried
  28 different amounts. Adding any amount of the experts made things worse. So the best "mix" was the
  generalist alone.
- **Drop-and-boost mix.** Before mixing, randomly throw away 90% of each model's changes and make the
  rest 10 times bigger. This works in some published research. Here it broke the models badly.
- **Keep-the-biggest-changes mix.** Keep only the largest 20% of each model's changes, and settle
  disagreements by majority vote. The result was worse than doing nothing.
- **Best-fit mix.** Blend each layer of the network so its outputs match all five models as closely as
  possible. This was better than the last two, but still far behind the generalist.
- **Learned mix.** Let the computer find 375 mixing amounts on its own, one for each part of each
  model. It learned these in about 7 minutes. This is the only method that beat the generalist.

## The one success: the learned mix

The learned mix scores **3.141**. The generalist scores **3.156**.

- The gain is real, not luck. We checked by resampling the course's scoring text 10,000 times. The gain was
  always between 0.014 and 0.015.
- It got better at math, code and history.
- It got **worse at medicine**.
- It stayed very close to the generalist. It kept about 98% of the generalist's changes and added only
  about 3% of each expert's.

The gain is small. The learned mix closes about 4% of the gap between the generalist (3.156) and the
best possible (2.78).

## Why mixing mostly failed

We measured how much each model changed from the starting model, and in which direction.

- **The changes are large.** Each model moved about half as far as the size of the starting model
  itself.
- **The changes point the same way.** All five models moved in a very similar direction.

Mixing works best when each model changes different things. Here the models changed the same things.
So adding the experts on top of the generalist counts the same change twice or more. The model
overshoots and breaks. One example: adding just 10% of each expert to the generalist took the score
from 3.13 to 5.35 on our held-out text.

## Second try: mixing only the most important changes

The first try mixed everything the experts changed. The second try starts from the generalist and
adds only each expert's most important changes. We measured each expert's change from the generalist,
and kept only its strongest directions. A setting called the **rank** says how many directions to
keep. A low rank keeps a few; a high rank keeps many. This adds no size to the model.

Measured from the generalist, the experts overlap much less than when measured from the starting
model. So we hoped they would interfere less.

| Method | Score | Better than the generalist? |
|---|---|---|
| Keep the strongest changes | 3.170 | No, worse by 0.015 |
| Keep the strongest changes, and stop them from overlapping | 3.160 | No, worse by 0.004 |

- Both were worse than the generalist on all four kinds of text.
- Stopping the changes from overlapping always helped. But it was not enough.
- Both picked the smallest amount we tried. In other words, the less we added, the better. Adding
  nothing at all would have been best.

## A check: what if each text got its own expert's changes?

We also asked a "what if" question. Suppose a model could always tell which kind of text it is
reading. It could then use only that expert's strongest changes. How well would that do?

This is only a check, not a model we can submit. It tells us whether such a model is worth building.

| What each kind of text gets | Score |
|---|---|
| The generalist, unchanged | 3.136 |
| The expert's strongest changes (any rank we tried) | 3.40 to 3.56. Worse. |
| The same, plus the expert's word list (its "embeddings") | 3.117 at best. Slightly better. |
| The expert's full model | 2.747 |

(These scores use our held-out text, so they differ a little from the other tables. They come from the
re-run after the bug fix.)

What this shows:

- **An expert's knowledge is spread across the whole model.** A small part of its changes does not
  carry its skill.
- **The changes only work together with the expert's own word list.** Without it, they make things
  worse.
- Even with perfect routing, the expert's word list and the largest rank the size limit allows, the
  pieces win back only about 5% of the gap. They help a little on math, code and history, and hurt
  medicine. A routed model would need training to make the pieces fit together.

## Things we did not test

- Several methods picked a setting at the very edge of what we tried. In the second try, the best
  choice may simply be "add nothing." We did not try settings in between.
- The learned mix always started from the generalist. We did not try other starting points.
- We ran each mixing method once more after the bug fix. The learned mix, the only one that trains,
  came out the same to within 0.00002. Each method still used one random seed.

## Cost

All 34 runs, including the error analysis and the re-runs, took about an hour of GPU time in
total. The learned mix was the most expensive at 7 minutes.

## Where to find more

- Each run has its own notes in `results/<run name>/notes.md`.
- The dashboard shows all runs with charts: run `./lab dashboard`.
- The code is in `training_scripts/merging/`.
