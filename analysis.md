# Analysis: where we stand

November 19, 2026. Everything we have measured so far, in one place: 39 runs, about 1.4 GPU-hours.
`results_summary.md` is the short version for readers new to the project. `error_analysis.md` has the
full detail behind section 5. Each number here comes from a run in `results/`; section 10 lists them.

Scores are next-token loss. **Lower is better.** A difference of 0.01 matters in this project.

## The short version

- **The bar is 3.156**: the generalist's dev score. The best we could hope for is **2.779**, if every
  text could use its own expert. The gap between them is 0.377.
- **We tried seven ways to mix the models' weights without training on new text** (Method 1).
- **One worked**: the learned mix scores **3.141**, 0.014 better than the generalist. That closes 4% of
  the gap.
- **The others were worse than the generalist**, most of them by a lot.
- **We know why.**
  - Each expert's skill is spread through the whole model and tied to its own word embeddings.
  - The experts forgot general ability while they learned their domain.
  - Their changes are so large and so alike that adding them overshoots.
- **What it points to** (our reading): a model needs different weights for different text, and pieces
  taken from the experts will need training to fit together.
- **Pruning opens a new route** (section 7, held-out slice, no training, perfect routing).
  - Each expert can lose 35% of its FFN neurons at no cost on its own text.
  - That frees room for 4 experts that keep their own attention and share only the word embeddings.
  - With a small per-expert correction to the shared embeddings, this scores **2.999 vs 3.136** for the
    generalist: 35% of the gap.
  - It is not yet a real model: it needs a router, and probably a ruling on whether it is allowed.

## 1. The task

We start from five provided GPT-2 small models (124M parameters each). All were trained from one shared
**starting model** (the "seed"):

- four **experts**, each trained further on one kind of text: math, code, medicine or history;
- one **generalist**, trained further on a mix of all four.

The goal is one model that scores well on all four kinds of text. It may have at most 300M parameters
in total and 150M in use for any one token. It must predict each word from the words before it only.

## 2. How we measure

- **Dev score.** The course's scoring text: 512 windows of 2,048 tokens per kind of text. We only
  compare models on it. We never use it to choose settings.
- **Held-out slice.** To choose settings, we set aside the last 2 million tokens of each training file.
  All tuning in this document used it.
- **Is a difference real?** We resample the dev windows 10,000 times, pairing the two models on the same
  windows, and report the range that holds 95% of the results. If the range does not include 0, the
  difference is real.
- **Our scoring matches the course's.** All provided models score exactly the numbers the course gave
  us.

## 3. The provided models

Dev scores:

| Model | Math | Code | Medicine | History | Mean |
|---|---|---|---|---|---|
| **Generalist** | 3.142 | 3.169 | 2.787 | 3.525 | **3.156** |
| Math expert | **2.675** | 3.266 | 3.446 | 3.937 | 3.331 |
| Code expert | 3.240 | **2.709** | 3.876 | 4.280 | 3.527 |
| Medicine expert | 3.574 | 3.833 | **2.586** | 3.872 | 3.466 |
| History expert | 3.514 | 3.760 | 3.337 | **3.145** | 3.439 |
| Starting model | 3.449 | 3.495 | 3.084 | 3.846 | 3.468 |
| *Each text gets its own expert (not one model)* | 2.675 | 2.709 | 2.586 | 3.145 | *2.779* |

What stands out:

- **Each expert beats the generalist on its own text** by 0.20 (medicine) to 0.47 (math).
- **Each expert is worse than the generalist on all other text.** Often it is worse than the starting
  model too. The code expert scores 3.876 on medicine; the starting model scores 3.084.
- **Medicine has the smallest gap.** The generalist is already closest to its expert there.

Two more reference points:

- **Plain average of the four experts' weights:** 3.516. That is worse than every model it came from
  except the code expert.
- **The course's example** (the generalist trained 6 more minutes on a mix): 3.163. It is better on
  math and code, worse on medicine and history, and worse overall by 0.007.

## 4. Method 1: mixing the weights

Every method here builds one model of the same size as the inputs (125M parameters). None trains on new
text, except the learned mix, which trains only its mixing amounts. Each picked its setting on the
held-out slice.

| Method | What it does | Setting it chose | Dev | vs generalist (95% range) |
|---|---|---|---|---|
| **Learned mix** | Learns 375 mixing amounts, one per part per model | Learned, 300 steps | **3.141** | **−0.0144 (−0.0149 to −0.0140)** |
| Simple mix | Starting model plus amounts of each model's change | All generalist, no expert | 3.156 | 0 (it is the generalist) |
| Strongest changes, no overlap | Generalist plus each expert's strongest directions, kept apart | Smallest amount tried | 3.160 | +0.0044 (+0.0042 to +0.0045) |
| Strongest changes | Generalist plus each expert's strongest directions | Smallest amount tried | 3.170 | +0.0145 (+0.0142 to +0.0148) |
| Best-fit mix | Matches each layer's outputs to all five models | α 0.5 | 3.445 | +0.2891 (+0.2857 to +0.2925) |
| Drop-and-boost | Drops 90% of each change, scales the rest ×10 | Nothing added | 3.468 | +0.3130 (it is the starting model) |
| Keep-the-biggest | Keeps the largest 20% of each change; majority vote on sign | λ 0.4 | 3.720 | +0.5641 (+0.5604 to +0.5677) |

Technical names, in table order: learned per-group λ; task arithmetic; TSV-M; low-rank SVD merge;
RegMean; DARE; TIES.

**The learned mix, per kind of text** (dev, vs the generalist):

| | Math | Code | Medicine | History | Mean |
|---|---|---|---|---|---|
| Change | −0.041 | −0.024 | **+0.026** | −0.018 | −0.014 |
| Windows where it is better | 100% | 99% | 1% | 97% | |

- It stays very close to the generalist: about 98% of the generalist's change and about 3% of each
  expert's.
- It trades medicine for the other three, on almost every window.
- It took 7 minutes. Run twice, it scored the same to within 0.00002.

## 5. Why mixing does not work

The full evidence is in `error_analysis.md`. The five findings:

1. **The experts are better across their whole domain, not on a few special texts.**
   - Each expert beats the generalist on 96 to 99% of its domain's dev windows.
   - It predicts 62 to 65% of individual tokens better, ordinary words included: " of", ".", ",".
   - The gain is largest at the start of a window and shrinks somewhat, but most of it lasts to the
     end.
2. **The experts paid for it by forgetting** (section 3). Any mix that brings in an expert's skill
   brings in its forgetting too.
3. **Their changes are large and alike.**
   - Each model's change from the starting model is 41 to 52% of the starting model's own size.
   - The changes point mostly the same way (similarity 0.64 to 0.79 overall).
   - So adding experts on top of the generalist counts the same change several times. Adding just 10% of
     each expert sent the held-out loss from 3.13 to 5.35.
   - That model is broken inside: its internal signals are about 40% too small. Adjusting its
     confidence fixes only 0.23 of the 2.22 extra loss.
4. **There is no wall between the models.** Blending any two in a straight line changes the loss
   smoothly. On the four-domain average, no bump rises above 0.04. The blend is simply a trade: moving toward one expert helps its
   domain and hurts the other three. Even a 10% step toward any expert gives no gain on average.
5. **Expertise is tied to each model's own word embeddings.**
   - Swap word embeddings between two models and both break (code: 4.93 and 4.60, against 2.99 and 2.53).
   - An expert given the generalist's MLP layers loses all of its lead.
   - Swapping layer norms changes almost nothing.

So any single set of weights between these models gains on one domain only by losing on others. The
learned mix found a small trade that helps on average.

## 6. What a perfect router could get

Suppose a model always knew which kind of text it was reading, and could use that expert's strongest
changes (as many as the size limit allows). We tested this directly. It is a check, not a model we can
submit. Held-out slice:

| Each kind of text gets | Math | Code | Medicine | History | Mean |
|---|---|---|---|---|---|
| The generalist | 3.077 | 3.116 | 2.807 | 3.545 | 3.136 |
| Expert's strongest changes only | | | | | 3.40 to 3.56 (worse) |
| The same, plus the expert's word embeddings and small parts | 3.024 | 3.080 | 2.845 | 3.518 | 3.117 |
| The full expert | 2.572 | 2.676 | 2.597 | 3.142 | 2.747 |

- **Pieces cut out of the experts recover only about 5% of the gap,** even with perfect routing.
- **They help only with their own word embeddings.**
- **Medicine gets worse again.**

## 7. Pruning the experts (first results)

Each expert has 3,072 FFN neurons in each of its 12 layers. We tested whether some can be removed while
the expert stays as good on its own kind of text. All numbers here are on the held-out slice. None
involve training.

**How much can be removed** (`20261119-1610`, `-1623`):

- Rank all of an expert's neurons together, by how much each one affects the loss on its own text.
  Remove the lowest, then refit each layer's output to make up for them.
- **Keeping 65%** of the neurons costs under 0.001 on the expert's own text. 60% costs about 0.012; 50%
  costs 0.08 to 0.09.
- The removed neurons sit mostly in layers 1, 2, 3 and 11, which keep only 5 to 18% of theirs. Removing
  the same share from every layer instead is much worse.
- The four experts tend to keep the same neurons.

**What that buys.** Under the 300M limit, removing 35% of each expert's FFN neurons leaves room for four
experts that each keep their own attention. They share only the word embeddings. We call this design A.
We compared it with Method 2 as planned (design B: four full FFNs, one shared attention). Each kind of
text goes to its own expert's parts, as if a perfect router existed (`20261119-1702`, `-1707`):

| Each kind of text gets | Math | Code | Medicine | History | Mean |
|---|---|---|---|---|---|
| The generalist | 3.077 | 3.116 | 2.807 | 3.545 | 3.136 |
| The learned mix | 3.035 | 3.093 | 2.835 | 3.526 | 3.122 |
| B: shared attention, own full FFN | 3.388 | 3.540 | 2.986 | 3.860 | 3.443 |
| A: shared embeddings, nothing else | 3.121 | 3.441 | 3.730 | 4.631 | 3.731 |
| **A: shared embeddings + a rank-128 correction per expert** | **2.798** | **2.894** | 2.861 | **3.442** | **2.999** |
| The full expert | 2.572 | 2.676 | 2.597 | 3.142 | 2.747 |

- **Shared embeddings are the whole problem.** Without a correction, design A is far worse than the
  generalist. Pruning is not the cause: the unpruned version scores the same.
- **A small correction fixes most of it.** Each expert gets its own low-rank change to the shared
  embeddings (the average of the experts'), paid for with more FFN neurons. Rank 96 to 128 is best: at
  rank 128 each expert keeps 52% of its neurons. Higher ranks cost too many neurons.
- **It recovers 35% of the gap**, against 4% for the learned mix. Of the 0.25 that remains, about 0.07
  comes from pruning and 0.19 from sharing the embeddings.
- **Medicine is still worse than the generalist** (2.861 vs 2.807). With the generalist's embeddings
  instead, medicine edges past it, but the other three get worse.
- **Design B is worse than the generalist without training.** Method 2 would rely entirely on its
  training step to repair the shared attention.

**Not yet a real model:**

- The routing used the true label. A real router guesses from the text so far and gets some early
  tokens wrong.
- There is no dev score, because the course's scoring script cannot run a routed model as is.
- Design A keeps each expert nearly whole and shares only embeddings. The limit exists to stop "a
  classifier which picks a provided expert model per document". This needs the same Piazza question as
  Method 2, and more urgently.

## 8. What this means for the next methods

This section is our interpretation. The sections above are measured.

- **Mixing weights is close to done.** The learned mix is probably near the best single trade, and every
  check after the bug fix confirmed the results.
- **Recovering more of the 0.377 gap needs different weights for different text.** The gap is broad
  and shows from a window's first tokens, so a router has early signal to act on.
- **Method 2 (mixture of experts built from expert parts) will depend on its training step.** Expert
  parts do not work without their own embeddings, and a shared model has one set of embeddings. That is
  the main tension. Section 7 shows a cheap partial answer: a small per-expert embedding correction.
- **Pruned experts with their own attention (design A) start from a much better place than Method 2**
  (2.999 vs 3.443 before any training). If the rules allow it, it is the strongest starting point we
  have.
- **Method 3 (teaching one model from the experts) avoids the compatibility problem.** It copies the
  experts' predictions, not their weights. The open question is room: the experts' skill is broad, and a
  124M model may not hold all four.
- **Watch medicine.** It is the domain every method so far has hurt.

## 9. How sure we are

- **A bug, fixed and checked.**
  - For math, our held-out slice was mostly one GMAT reading guide, not math.
  - We fixed it on November 18 and re-ran all 12 affected runs on November 19.
  - Every merge chose the same setting and scored the same on dev.
  - The diagnostics' conclusions held. One detail changed: the gain is not flat across the window.
- **Dev differences are solid.** The 95% ranges above are narrow, about ±0.001 or less.
- **Reproducible.** The learned mix scored the same to within 0.00002 when re-run. It is not
  bit-identical, because GPU arithmetic varies slightly between runs.
- **Not yet tested:**
  - Several searches chose the smallest amount we tried, so settings below it are untested.
  - The learned mix always started from the generalist.
  - Every run used one random seed.
  - The diagnostics used 16 to 32 held-out windows per domain: enough for the large effects here, not
    for small ones.

## 10. Where the numbers come from

| What | Run in `results/` |
|---|---|
| Provided models | `20261118-1942` to `-1955` (`baseline_*`) |
| Mixing methods | `20261119-0502` to `-0527` (`merging_*`; re-runs after the fix) |
| Size and overlap of the changes | `20261118-2003_merging_tv-stats_hz` |
| Perfect-router check | `20261119-0535_merging_routed-oracle_hz` |
| Swapping parts, calibration, token gains, straight-line blends | `20261119-1535` to `-1538` (`analysis_*`) |
| Training files are not ordered | `20261118-2129_analysis_slices_hz` |
| Pruning sweeps | `20261119-1610_pruning_sweep_hz`, `20261119-1623_pruning_sweep-fine_hz` |
| Designs A and B with perfect routing | `20261119-1702_pruning_assemble-oracle_hz`, `20261119-1707_pruning_assemble-oracle-ranks_hz` |
| Per-window dev comparisons | `tools/analysis/dev_windows.py`; ranges from `lib/bootstrap.py` |

The code is in `training_scripts/merging/`, `training_scripts/pruning/` and `training_scripts/analysis/`. Each run's `notes.md` says
what it showed. `./lab dashboard` shows all runs with charts.
