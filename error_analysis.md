# Error analysis: why merging the experts did not work

November 18, 2026; updated November 19 after re-running the affected runs. Covers the 17 scored runs
(8 baselines, 9 merges and checks), their 8 re-runs, and 5 diagnostic runs.
Scores are next-token loss in nats; **lower is better**. "Dev" numbers are official course scores. The
diagnostics use our held-out slice of the training pool instead, so they differ a little from dev.

## The short answer

1. **The experts are better almost everywhere in their own domain, not in a few special documents.**
   What they know is a broad shift in how the domain's text looks.
2. **They paid for it by forgetting.** Away from its own domain, each expert is often worse than the
   untrained starting model.
3. **Their changes are large and point the same way.** Adding them together counts the same change
   several times, so the model overshoots and breaks.
4. **The models are not in "different valleys".** Moving from the generalist toward an expert is smooth.
   It simply trades one domain against the other three.
5. **Expertise is spread through the whole model and tied to its own word embeddings.** You cannot lift
   out a small piece of an expert, or swap a part between models, and keep the skill.

Together these explain the result: any single set of weights between the generalist and the experts
gains on one domain only by losing on others. The best we found, the learned mix, made a small trade
that helps on average (3.141 vs 3.156).

## 1. First, a bug in our own measuring

Before the findings: one of our tools was wrong, and it affected some numbers.

To pick settings without touching dev, we set aside the last 2 million tokens of each pool file
("pool-val"). The scoring code then used the **first** 64 windows of that slice, in a row. For math,
79% of those windows are **one 104,000-token document: a GMAT reading-comprehension guide**. It is
English prose, not math, and the math expert is barely better than the generalist on it (a 0.02 to
0.04 lead, against 0.47 on dev).

- **Found by:** the `swaps` diagnostic showed the math expert barely beating the generalist. The
  `slices` diagnostic then showed the pool file is not ordered (the lead is 0.40 to 0.55 everywhere in
  `pool/math.bin`). Reading the windows found the book.
- **Fixed in:** commit `d725d3f`. Windows are now spread over the whole slice; they touch at least 149
  documents per domain, and no document holds more than 6%.
- **Not affected:** every dev score, and every comparison between runs on dev.
- **Affected:** every setting chosen on pool-val before the fix (λ, rank, α, and the learned mix's
  monitoring), and the math column of every pool-val table. Each affected run's notes carry a caveat.
- **Re-run on November 19 with the fix.** All 8 affected runs were repeated with the same grids. Every
  merge picked **exactly the same setting**, and its dev per-window losses are identical to the original.
  On the fixed windows the merges are still worse than the generalist in every domain, so fixing math
  did not flip any choice. The learned mix's training never reads pool-val; it reproduced to within
  0.00002 on dev (not bit-identical: individual windows differ by up to 0.005, from GPU arithmetic).
  **Every Method 1 result stands.**
- **One conclusion changed:** the perfect-routing check in section 6, which now has valid math numbers.

## 2. What the experts are better at

**The gap covers the whole domain.** On dev, each expert beats the generalist on 96 to 99% of its
domain's 512 windows. The 10% of windows with the biggest gaps hold only 16 to 20% of the total gap.

| Domain | Generalist | Own expert | Gap | Expert better on |
|---|---|---|---|---|
| math | 3.142 | 2.675 | 0.467 | 98% of windows |
| code | 3.169 | 2.709 | 0.459 | 99% |
| medicine | 2.787 | 2.586 | 0.200 | 97% |
| history | 3.525 | 3.145 | 0.380 | 96% |

**It covers most words, too** (diagnostic `tokens`; code, medicine and history, since math used the
biased windows):

- The expert predicts 60 to 65% of individual tokens better.
- Per occurrence, the gain is largest on rare tokens. In code, the rarest fifth of tokens gains 0.97
  nats each and the commonest fifth 0.11.
- In total, though, ordinary tokens carry the most gain because they are so frequent. The top tokens by
  total gain are things like `.`, ` the`, `,` and newlines. The 100 best token types hold only 19 to 21%
  of the gain. So the expert is better at ordinary words in a domain context, not just at jargon.
- The gain is about the same at every position in a 2,048-token window (code: 0.49, 0.49, 0.53 and 0.44
  across four position ranges). It is not a first-guess advantage that context soon erases.

**The generalist is already part of the way there.** Measured from the starting model, the generalist
already has 40 to 60% of each expert's gain on that expert's own domain: math 40%, code 42%, medicine
60%, history 46%.

## 3. What the experts gave up

Away from its own domain, each expert is worse than the generalist, and **often worse than the
untrained starting model** (dev):

| Expert | Its worst other domain | Expert | Starting model | Generalist |
|---|---|---|---|---|
| code | medicine | 3.876 | 3.084 | 2.787 |
| code | history | 4.280 | 3.846 | 3.525 |
| math | medicine | 3.446 | 3.084 | 2.787 |
| medicine | code | 3.833 | 3.495 | 3.169 |

So an expert's change is not "new skill added to the starting model". It is new skill **plus** a large
loss of general ability. Any merge that brings in the skill brings in the forgetting with it.

## 4. Why adding the experts' changes overshoots

Measured from the starting model, each model's change is large: 41 to 52% of the size of the starting
model's own weights. The changes also point mostly the same way: cosine similarity is 0.64 to 0.79
overall, and above 0.99 in the layer norms (diagnostic `tv-stats`).

A large part of every change is therefore **shared**: the general drift of "more training". The
generalist already contains it. Adding the experts' changes on top adds that shared part again, once per
expert:

- Adding just 10% of each expert to the full generalist moved the held-out loss from 3.19 to 5.36.
- In that overshooting model, the internal signals (the hidden state after each block) are about 40%
  smaller than normal in every block (diagnostic `scale`).
- Re-scaling the model's output confidence ("temperature") repairs only 0.22 of its 2.18 extra loss. The
  model is broken inside, not just overconfident.
- The plain average of the experts and the learned mix have normal internal signal sizes, and their best
  temperature is 1.0. Their loss is not a calibration problem either.

Tricks built to reduce interference (TIES, DARE) did not help, because the problem is not small,
conflicting edits. DARE throws away 90% of each change and multiplies the rest by 10. On changes this
large, that alone took the generalist from 3.19 to 20.2.

## 5. It is not a "different valleys" problem

A classic reason merges fail is that two models sit in different valleys of the loss landscape, with a
hill between them. Then any blend of the two is bad. We checked by walking in a straight line from one
model to the other (diagnostic `paths`):

- **There is no real hill.** The largest bump above a straight line on the 4-domain average is 0.039
  (generalist to code), 0.008 (math expert to code expert) and 0.002 (medicine expert to history
  expert).
- **The walk is a smooth trade.** From the generalist toward an expert, that expert's domain improves
  steadily (code 3.31 to 2.84, history 3.51 to 3.07, medicine 2.81 to 2.60). The 4-domain average gets
  steadily worse (3.21 to between 3.40 and 3.67). The average includes the biased math windows; it rises
  the same way without them.

So the models are compatible enough to blend. The trouble is that blending cannot give one domain more
without giving the others less.

## 6. Expertise is spread through the model and tied to its word embeddings

**Swapping parts** (diagnostic `swaps`, own-domain held-out loss; math left out):

| Change | Result |
|---|---|
| Generalist with the expert's word embeddings | Far worse than either model (code 5.45; generalist 3.31, expert 2.84) |
| Expert with the generalist's word embeddings | Far worse than either model (code 5.16) |
| Swap the layer norms, either way | Almost no change (under 0.02) |
| Generalist with the expert's attention, or the expert's MLP | Worse than the generalist |
| Expert with the generalist's attention | Keeps 35 to 52% of its lead |
| Expert with the generalist's MLP | Keeps none of its lead |

Each model's body only works with its own word embeddings. Averaging embeddings across models mixes
two incompatible "dictionaries".

**Low-rank pieces** (diagnostic `routed-oracle`, held-out loss): we gave each domain only its own
expert's strongest changes (rank 16 to 160), as if a perfect router existed.

Numbers below are from the re-run with fixed windows (`20261119-0535_merging_routed-oracle_hz`).

| Each domain gets | math | code | medicine | history | mean |
|---|---|---|---|---|---|
| The generalist (no piece) | 3.077 | 3.116 | 2.807 | 3.545 | 3.136 |
| Rank 160 piece, generalist's other parts | 3.548 | 3.635 | 3.064 | 3.978 | 3.557 |
| Rank 160 piece, expert's embeddings and small parts | 3.024 | 3.080 | 2.845 | 3.518 | 3.117 |
| The expert itself | 2.572 | 2.676 | 2.597 | 3.142 | 2.747 |

- With the generalist's embeddings and other small parts, every rank is worse than the generalist, in
  every domain.
- With the expert's own embeddings and small parts, rank 160 beats the generalist on math, code and
  history, but not on medicine. On average it wins back only about 5% of the gap to the experts (math
  10%, code 8%, history 7%, medicine −18%).
- **Correction history:** the first version of this check said it "never beats the generalist"; that
  came from the math bug. The table above is the corrected result.

In the weights themselves, keeping each expert's top 160 directions holds only 51 to 69% of the "energy" of its change,
depending on the layer type. The changes are not low-rank.

## 7. Medicine loses every time

- It has the smallest gap to its expert (0.200), and the generalist is closest to the expert there (60%
  of the way).
- Every failed merge hurt medicine the most (for example, the expert average: +0.61 on medicine, +0.17
  on math).
- The learned mix improved math, code and history but made **99% of medicine windows** worse. It hurt
  most exactly where the medicine expert helps most (correlation −0.60 between its improvement on a
  window and the medicine expert's advantage there).

One reading is that medicine depends most on the general ability the generalist has and the other
experts lost. That is our interpretation, not something we tested directly.

## 8. What this means (interpretation)

The facts above are measured. This section is our reading of them.

- **One dense set of weights between these models cannot get far below the generalist.** Each expert's
  skill comes bundled with forgetting, the changes overlap heavily, and every step toward one domain is
  a step away from the others. The learned mix probably sits near the best trade, about 0.014 better.
  Re-running every selection with the fixed held-out windows chose the same settings.
- **To recover much of the 0.38 gap, a model has to use different weights for different text.** The
  per-window gap is broad and present from the first tokens of a window, which is what routing on the
  text so far needs.
- **Pieces taken from the experts will need training to fit together.** Their skill depends on their own
  embeddings and on attention and MLP working together. So a mixture-of-experts built from expert parts
  (the plan's Method 2) will lean on its training step, and shared embeddings will be the tension point.
- **Teaching one model from the experts (distillation, the plan's Method 3)** avoids the compatibility
  problem entirely: it copies the experts' predictions, not their weights. The token analysis says what
  it has to learn is broad, so a 124M dense model may run out of room. That is the capacity question the
  plan already raises.

## 9. Limits of this analysis

- The diagnostics used 32 held-out windows per domain (16 for tokens). That is enough to see the large
  effects here, not small ones.
- Except for `slices`, every diagnostic ran before the fix, so its math numbers describe the GMAT book.
  Conclusions above leave math out wherever that matters.
- The merges were re-run once after the fix, with the same grids. Several searches still stop at the
  edge of their grid, so settings beyond it are untested.
- Every run used one seed.

## Where the numbers come from

| Finding | Run (in `results/`) or script |
|---|---|
| Gap per window, off-domain loss, learned mix per window | `tools/analysis/dev_windows.py` (dev window losses of all runs) |
| Task-vector sizes and overlap | `20261118-2003_merging_tv-stats_hz` |
| Pool files not ordered | `20261118-2129_analysis_slices_hz` |
| Swapping parts | `20261118-2127_analysis_swaps_hz` |
| Token-level gains | `20261118-2130_analysis_tokens_hz` |
| Calibration and signal sizes | `20261118-2130_analysis_scale_hz` |
| Straight-line paths | `20261118-2131_analysis_paths_hz` |
| Low-rank pieces with perfect routing | `20261119-0535_merging_routed-oracle_hz` (re-run; replaces `20261118-2105`) |
| Re-runs after the fix | `results/20261119-*_merging_*` (each lists the run it replaces as its parent) |

The diagnostics code is `training_scripts/analysis/diagnostics.py`.
