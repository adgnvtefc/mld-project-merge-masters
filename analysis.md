# Analysis: where we stand

November 19, 2026. Everything we have measured so far, in one place: 62 runs (57 by Hanmo, 5 by Ayush),
about 12.0 GPU-hours.
`results_summary.md` is the short version for readers new to the project. `error_analysis.md` has the
full detail behind section 5. Each number here comes from a run in `results/`; section 14 lists them.

Scores are next-token loss. **Lower is better.** A difference of 0.01 matters in this project.

## The short version

- **The bar is 3.156**: the generalist's dev score. The best we could hope for is **2.779**, if every
  text could use its own expert. The gap between them is 0.377.
- **We tried seven ways to mix the models' weights without training on new text** (Weight Mixing).
- **One worked**: the learned mix scores **3.141**, 0.014 better than the generalist. That closes 4% of
  the gap.
- **The others were worse than the generalist**, most of them by a lot.
- **We know why.**
  - Each expert's skill is spread through the whole model and tied to its own word embeddings.
  - The experts forgot general ability while they learned their domain.
  - Their changes are so large and so alike that adding them overshoots.
- **What it points to** (our reading): a model needs different weights for different text, and pieces
  taken from the experts will need training to fit together.
- **Two routed designs now beat everything else.** Both send each text to one expert's parts, chosen
  from the text read so far.
  - **Best so far: Pruned Expert Stacks with own rows for each expert's top 8,192 tokens, trained 50
    minutes at a lower learning rate with distillation from each stack's own expert, 2.899 on dev**
    (sections 9 to 11). It is 0.256 better than the generalist and closes 68% of the gap. Each expert
    keeps its own attention (sections 7 and 8).
  - **Close behind: Ayush's Shared-Attention MoE, trained the same way, 2.989.** Its experts share
    attention. It closes 44% of the gap.
  - **Untrained, Stacks was far ahead** (3.042 against 3.207). Training closed most of that difference.
  - **Training the generalist the same way gains only 0.037.** So the gain comes from routing to
    experts, not from extra training alone.

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

## 4. Weight Mixing

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
   smoothly. On the four-domain average, no bump rises above 0.04. The blend is simply a trade: moving
   toward one expert helps its domain and hurts the other three. Even a 10% step toward any expert gives no gain on average.
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
experts that each keep their own attention. They share only the word embeddings. This is **Pruned Expert
Stacks** (Stacks for short). We compared it with an untrained Shared-Attention MoE without embedding maps
(four full FFNs, one shared attention). Each kind of
text goes to its own expert's parts, as if a perfect router existed (`20261119-1702`, `-1707`):

| Each kind of text gets | Math | Code | Medicine | History | Mean |
|---|---|---|---|---|---|
| The generalist | 3.077 | 3.116 | 2.807 | 3.545 | 3.136 |
| The learned mix | 3.035 | 3.093 | 2.835 | 3.526 | 3.122 |
| MoE (untrained, no maps): shared attention, own full FFN | 3.388 | 3.540 | 2.986 | 3.860 | 3.443 |
| Stacks: shared embeddings, nothing else | 3.121 | 3.441 | 3.730 | 4.631 | 3.731 |
| **Stacks: shared embeddings + a rank-128 fix per expert** | **2.798** | **2.894** | 2.861 | **3.442** | **2.999** |
| The full expert | 2.572 | 2.676 | 2.597 | 3.142 | 2.747 |

- **Shared embeddings are the whole problem.** Without a correction, Stacks is far worse than the
  generalist. Pruning is not the cause: the unpruned version scores the same.
- **A small correction fixes most of it.** Each expert gets its own low-rank change to the shared
  embeddings (the average of the experts'), paid for with more FFN neurons. Rank 96 to 128 is best: at
  rank 128 each expert keeps 52% of its neurons. Higher ranks cost too many neurons.
- **It recovers 35% of the gap**, against 4% for the learned mix. Of the 0.25 that remains, about 0.07
  comes from pruning and 0.19 from sharing the embeddings.
- **Medicine is still worse than the generalist** (2.861 vs 2.807). With the generalist's embeddings
  instead, medicine edges past it, but the other three get worse.
- **The MoE is worse than the generalist without training.** It relies entirely on training to repair
  the shared attention.

These numbers used the true label to route. Section 8 replaces it with a real router.

## 8. Stacks as a real model, with a router

`training_scripts/routed/build.py` builds Stacks from the provided models in about a minute and a
half, with no training, as a solution the course's script can score (`20261119-1845`, `-1850`; the
deterministic reference build is `20261119-1936`).

**The router has no parameters.** It uses the experts themselves:

- For the first *k* tokens of a text, all four experts run. The model's prediction mixes theirs,
  weighted by how well each one predicted the text so far.
- At token *k*, the text commits to the expert that predicted it best. That expert predicts the rest.
- Every expert has its own attention, so the chosen expert re-reads the first *k* tokens. Running four
  experts at the start costs extra compute: active parameters grow with *k*. The limit allows *k* up to
  about 232.

**How often it picks the right expert** (held-out slice, share of texts):

| After this many tokens | 1 | 8 | 32 | 128 | 511 |
|---|---|---|---|---|---|
| math | 42% | 75% | 89% | 94% | 92% |
| code | 58% | 86% | 86% | 92% | 98% |
| medicine | 50% | 86% | 94% | 97% | 100% |
| history | 48% | 80% | 92% | 98% | 100% |

**Dev scores** (official script; *k* chosen on the held-out slice):

| Model | Math | Code | Medicine | History | Mean | Active | vs generalist (95% range) |
|---|---|---|---|---|---|---|---|
| Generalist | 3.142 | 3.169 | 2.787 | 3.525 | 3.156 | 123.5M | |
| Learned mix | 3.100 | 3.145 | 2.813 | 3.507 | 3.141 | 123.5M | −0.014 |
| Stacks, *k* = 128 | 2.915 | 2.954 | 2.859 | 3.458 | 3.047 | 129.1M | −0.109 (−0.116 to −0.102) |
| **Stacks, *k* = 224** | **2.913** | **2.947** | 2.853 | **3.452** | **3.041** | 148.5M | **−0.114 (−0.121 to −0.107)** |
| Stacks, *k* = 224, deterministic build | 2.913 | 2.948 | 2.853 | 3.453 | 3.042 | 148.5M | −0.114 (−0.121 to −0.107) |

- **The real router costs about 0.013** against the true label on the held-out slice (3.012 vs 2.999).
- **Better on 91 to 94% of math and code windows, 76% of history, but only 17% of medicine.** Medicine is
  0.066 worse than the generalist.
- **Both models pass the official checks:** 299,995,804 parameters, causal, and active parameters under
  150M. At *k* = 224 the margin is only 1%.
- **Later commit points are slightly better.** On 256 held-out windows per domain, the loss falls steadily
  from *k* = 128 to 224 (3.050 to 3.043). On 64 windows this was hidden by single misrouted texts.
- **The build is deterministic.** Two builds gave bit-identical weights and the same dev score. An earlier
  version pruned slightly different neurons on each build, which flipped a near-tied routing decision.
- **On the rules:** each expert is pruned, refit and given new shared embeddings, so none is used as
  provided. We read the 300M limit as stopping routing to the provided experts as they are, so we
  consider this allowed.

## 9. Training the routed models (Stacks and MoE)

Ayush built the Shared-Attention MoE and trained it (`training_scripts/moe/`).

- **Shared parts:** word and position embeddings, attention and layer norms, all starting from the
  average of the five models.
- **Experts:** each expert keeps its full FFN layers, so the model has 297.6M parameters.
- **Router:** a linear classifier on the average of the word embeddings read so far. It picks an expert
  for each token. It is fitted in a closed form, using no gradient training, on training text.
- **Embedding maps:** a small map per expert (768 × 768) translates the shared embeddings into that
  expert's own. It is his answer to the embedding problem, where Stacks uses low-rank fixes.
- **Active parameters:** each expert's matrices run only on the tokens sent to it, so the official
  counter sees 124.7M.
- **Training:** next-token loss on the training text, 800 steps of 32 texts. That is 52M
  tokens, equal across domains, with each text sent to its own expert, in about 25 minutes.

We then trained Stacks the same way (`training_scripts/routed/heal.py`): the same data, in the same
order, the same steps and the same schedule. So the Trained Generalist control applies to it too. "FFNs
frozen" means the experts' FFN layers do not train; everything else does.

**Dev scores** (`20261119-0302`, `-0352`, `-0423`, `-0450`, `-2206`, `-2241`):

| Model | Training | Math | Code | Medicine | History | Mean | vs generalist (95% range) |
|---|---|---|---|---|---|---|---|
| Generalist | none | 3.142 | 3.169 | 2.787 | 3.525 | 3.156 | |
| MoE, untrained | none | 3.051 | 3.168 | 2.968 | 3.640 | 3.207 | +0.051 (+0.047 to +0.056) |
| Trained Generalist (control) | 0.40 GPU-h | 3.056 | 3.087 | 2.822 | 3.510 | 3.119 | −0.037 (−0.039 to −0.035) |
| Stacks, untrained | none | 2.913 | 2.948 | 2.853 | 3.453 | 3.042 | −0.114 (−0.121 to −0.107) |
| MoE, all trained | 0.47 GPU-h | 2.887 | 2.932 | 2.801 | 3.415 | 3.009 | −0.147 (−0.153 to −0.141) |
| Stacks, all trained | 0.56 GPU-h | 2.879 | 2.906 | 2.791 | 3.392 | 2.992 | −0.164 (−0.171 to −0.156) |
| MoE, FFNs frozen | 0.42 GPU-h | 2.865 | 2.916 | 2.781 | 3.395 | 2.989 | −0.166 (−0.172 to −0.160) |
| Stacks, FFNs frozen | 0.53 GPU-h | 2.858 | 2.887 | 2.772 | 3.375 | 2.973 | −0.183 (−0.190 to −0.175) |
| Stacks, all trained, 3× longer | 1.59 GPU-h | 2.870 | 2.897 | 2.780 | 3.380 | 2.982 | −0.174 (−0.181 to −0.166) |
| **Stacks, FFNs frozen, 3× longer** | 1.48 GPU-h | **2.849** | **2.878** | **2.761** | **3.364** | **2.963** | **−0.193 (−0.200 to −0.185)** |

GPU-hours include building the model and scoring it on dev. "3× longer" = 2,400 steps (157M tokens,
about 90 minutes), with the same warmup and the decay stretched to match.

- **Training does most of the work.** It takes the MoE from 3.207 to 2.989. The same
  training on the generalist gains only 0.037, and medicine gets worse for it.
- **For the MoE, freezing the FFNs beat training everything,** by 0.020 on every domain. Only the shared parts
  trained (71M of 298M parameters). These runs used 25 to 30 minutes of the 2-hour budget, so the order
  could flip with longer training.
- **Training fixes medicine.** With the experts frozen, medicine is level with the generalist (−0.006,
  range −0.014 to +0.003). Untrained, it was 0.181 worse.
- **Trained Stacks is the best model,** 0.016 better than the trained MoE (95% range
  0.013 to 0.020). It is better on 72% of math windows, 87% of code, 63% of medicine and 81% of history.
- **Its head start mostly disappeared.** Untrained, Stacks led by 0.165; trained, by 0.016. Training
  gained Stacks 0.069 and the MoE 0.218. Shared attention, trained on all four domains,
  seems to learn much of what Stacks' separate attention already had.
- **Medicine finally beats the generalist** with trained Stacks (−0.015, on 65% of windows).
- **The same pattern in both designs:** freezing the FFN layers wins at 800 steps. Training everything
  first makes the model worse (Stacks' held-out loss rose by 0.09 in the first 100 steps), then
  improves faster. At the end it was still gaining 0.014 per 100 steps, against 0.003 to 0.005 frozen.
- **Three times the training buys only 0.010,** for both variants (95% ranges 0.008 to 0.012 frozen,
  0.007 to 0.013 all trained). FFNs frozen still wins by 0.019. All trained lost 0.11 early and only
  recovered its starting loss after about 1,700 of 2,400 steps.
- **Routing costs both designs about the same:** 0.012 to 0.013 over the true label, on the held-out
  slice.

## 10. Fixing the shared embeddings in Stacks

Where sharing the embeddings costs, before training (`20261120-0618`, held-out, perfect routing):

| Each stack's embeddings | Mean | Recovered |
|---|---|---|
| Shared + rank-128 fix (as built) | 3.019 | |
| Own tables, input side only | 3.000 | 0.018 |
| Own word table, output side only | 2.863 | 0.156 |
| Own tables, both sides (ceiling, over budget) | 2.832 | 0.187 |
| Own rows for the top 4,096 / 16,384 tokens | 2.928 / 2.859 | 0.090 / 0.160 |

- **The cost is on the output side,** where the tied embedding table scores every next token.
- **Own rows for frequent tokens are the cheap fix.** The top 4,096 tokens cover 80 to 88% of the text.

Splitting the same 300M differently (`20261120-0628`, before training, with the router):

| Fix rank : own rows | FFN kept | Held-out |
|---|---|---|
| 128 : 0 (before) | 52.4% | 3.025 |
| 64 : 4,096 | 52.9% | 2.969 |
| **32 : 8,192** | 50.4% | **2.963** |
| 0 : 8,192 | 53.4% | 3.043 |

A small fix is still needed for rare tokens. Trained the same 30 minutes (`20261120-0750`), the 32 : 8,192
split scores **2.957** on dev: 0.016 better than the rank-128 split (95% range 0.014 to 0.018) and 0.006
better than that split trained 90 minutes. Training kept about a quarter of the untrained lead.

## 11. The learning rate was too high

Every training run first got worse, then recovered: Stacks + own rows by 0.038, Stacks all trained by
0.11, and even the plain generalist control by 0.083. These models are already trained, and a 3e-4 peak
after a 30-step warmup knocks them off their solution. Same model, same data and steps, peak 1e-4 after
a 200-step warmup (`20261120-1715`, `-1753`):

| Stacks + own rows, 30 min | Learning rate | Dev | vs 3e-4 (95% range) |
|---|---|---|---|
| FFNs frozen | 3e-4 | 2.957 | |
| FFNs frozen | 1e-4 | 2.929 | −0.028 (−0.030 to −0.027) |
| All trained | 1e-4 | 2.927 | −0.030 (−0.031 to −0.029) |
| All trained, FFNs at 0.3× the rate | 1e-4 | 2.924 | −0.033 |
| The same, 2× longer (1,600 steps, ~70 min) | 1e-4 | 2.918 | −0.039 |
| **The same, 800 steps, with distillation (~50 min)** | **1e-4** | **2.899** | **−0.058** |

- **The early damage is gone with FFNs frozen.** Held-out loss falls at every validation.
- **All trained now wins, narrowly** (by 0.0018, 95% range 0.0012 to 0.0025). It still rose a little as
  the learning rate reached its peak; training the FFNs at 0.3× the rate removes that rise and gains
  another 0.0027 (95% range 0.0020 to 0.0033).
- **Distillation from each stack's own unpruned expert is the biggest step since the learning rate.**
  Loss = half next-token, half matching the expert's full prediction. It gains 0.025 over the same recipe
  without it (95% range 0.024 to 0.025) and beats twice the training without it by 0.018. It trains about
  30% slower.
- **This changes earlier conclusions.** "FFNs frozen beats all trained" and "3× longer buys only 0.010"
  were measured at the too-high learning rate.

## 12. What this means for the next methods

This section is our interpretation. The sections above are measured.

- **Mixing weights is close to done.** The learned mix is probably near the best single trade, and every
  check after the bug fix confirmed the results.
- **Recovering more of the 0.377 gap needs different weights for different text.** The gap is broad
  and shows from a window's first tokens, so a router has early signal to act on.
- **Training is what makes routed experts work.** Section 9 confirms what the error analysis
  predicted: expert parts need training to fit together. 25 minutes took the MoE from worse than the
  generalist to the best model we have.
- **Trained Stacks is the front-runner, but narrowly.** Its lead over the trained MoE
  (0.016) is a fraction of its untrained lead. Either design could win with a different training plan.
- **Longer training helps at the lower learning rate:** twice the steps gained 0.0067, and the loss was
  still falling. The final recipe can afford about 1,600 to 2,000 steps within 2 hours.
- **Stacks' active parameters are 1% under the limit** (148.5M, from its commit point of 224
  tokens). A final recipe should either keep that margin checked or commit earlier (128: 129.1M).
- **Distillation (teaching one model from the experts) avoids the compatibility problem.** It copies the
  experts' predictions, not their weights. The open question is room: the experts' skill is broad, and a
  124M model may not hold all four.
- **Watch medicine.** Every untrained method hurt it. Training brought it level with the generalist
  (MoE) or just past it (Stacks, by 0.015).

## 13. How sure we are

- **A bug, fixed and checked.**
  - For math, our held-out slice was mostly one GMAT reading guide, not math.
  - We fixed it on November 18 and re-ran all 12 affected runs on November 19.
  - Every merge chose the same setting and scored the same on dev.
  - The diagnostics' conclusions held. One detail changed: the gain is not flat across the window.
- **Dev differences are solid.** The 95% ranges above are narrow: about ±0.001 for the mixes and
  ±0.006 to ±0.007 for the routed models.
- **Our two hosts agree.** Ayush's machine scores the generalist identically, window for window.
- **Reproducible.** The learned mix scored the same to within 0.00002 when re-run. It is not
  bit-identical, because GPU arithmetic varies slightly between runs. Stacks' build is now
  bit-identical from run to run. The training runs in section 9 have not been repeated yet, and GPU
  training is not bit-identical; the final recipe must reproduce within 0.005.
- **Not yet tested:**
  - Several searches chose the smallest amount we tried, so settings below it are untested.
  - The learned mix always started from the generalist.
  - Every run used one random seed and one learning rate.
  - The diagnostics used 16 to 32 held-out windows per domain: enough for the large effects here, not
    for small ones.

## 14. Where the numbers come from

| What | Run in `results/` |
|---|---|
| Provided models | `20261118-1942` to `-1955` (`baseline_*`) |
| Mixing methods | `20261119-0502` to `-0527` (`merging_*`; re-runs after the fix) |
| Size and overlap of the changes | `20261118-2003_merging_tv-stats_hz` |
| Perfect-router check | `20261119-0535_merging_routed-oracle_hz` |
| Swapping parts, calibration, token gains, straight-line blends | `20261119-1535` to `-1538` (`analysis_*`) |
| Training files are not ordered | `20261118-2129_analysis_slices_hz` |
| Pruning sweeps | `20261119-1610_pruning_sweep_hz`, `20261119-1623_pruning_sweep-fine_hz` |
| Stacks and the untrained MoE with perfect routing | `20261119-1702_pruning_assemble-oracle_hz`, `20261119-1707_pruning_assemble-oracle-ranks_hz` |
| Stacks with the router, dev scores | `20261119-1845_routed_design-a_hz`, `20261119-1850_routed_design-a-commit_hz` |
| Stacks, deterministic builds | `20261119-1912`, `-1921` (bit-identical twins), `-1936` (reference) (`routed_design-a-det_hz`) |
| Shared-Attention MoE (Ayush) | `20261119-0302_moe_init-check_ay`, `20261119-0352_moe_heal-all_ay`, `20261119-0423_moe_heal-shared_ay` |
| Trained Generalist (control) and host check (Ayush) | `20261119-0450_dense_heal-ce_ay`, `20261119-0259_baseline_generalist_ay` |
| Stacks trained | `20261119-2206_routed_heal-frozen-ffn_hz`, `20261119-2241_routed_heal-all_hz` |
| Stacks trained 3× longer | `20261120-0033_routed_heal-frozen-ffn-2400_hz`, `20261120-0205_routed_heal-all-2400_hz` |
| Stacks at a lower learning rate | `20261120-1715_routed_heal-frozen-ffn-rows8k-lr1e-4_hz`, `20261120-1753_routed_heal-all-rows8k-lr1e-4_hz`, `20261121-0426_routed_heal-all-rows8k-ffnlr03_hz`, `20261121-0506_routed_heal-all-rows8k-ffnlr03-1600_hz`, `20261121-0619_routed_heal-all-rows8k-ffnlr03-kd05_hz` (distillation) |
| Stacks embeddings: where sharing costs, budget splits, trained best split | `20261120-0618_routed_embed-check_hz`, `20261120-0628_routed_embed-grid_hz`, `20261120-0750_routed_heal-frozen-ffn-rows8k_hz` |
| Per-window dev comparisons | `tools/analysis/dev_windows.py`; ranges from `lib/bootstrap.py` |

The code is in `training_scripts/`: `merging/`, `pruning/`, `routed/`, `moe/`, `dense/` and `analysis/`.
Each run's `notes.md` says what it showed. `./lab dashboard` shows all runs with charts.
