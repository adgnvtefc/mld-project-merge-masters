"""
Phase 2 check, no training: how well do routed designs built from pruned experts do when every text is routed to its
own domain's parts (oracle routing by the domain label)?

Design A: shared word and position embeddings; each expert keeps its own attention, LayerNorms and final LayerNorm,
plus the share of its FFN neurons that fits the 300M total (global taylor ranking on its own pool train text,
least-squares refit; see prune_sweep.py). Optionally each expert also gets a rank-r correction of the shared
embeddings toward its own (truncated SVD of the difference), paid for with FFN neurons.

Design B (Method 2 in the plan, for comparison): shared embeddings, attention and block LayerNorms from one source;
each expert keeps its full FFN and its own final LayerNorm.

Routing by label means domain d is scored with the parameters of its own expert's route, so each route is a plain
GPT and is scored as one. Everything is scored on pool-val, never on dev. No model is saved; launch with --no-eval.

    python3 training_scripts/pruning/assemble_oracle.py --learned runs/<learned-lambda run>/model.bin --out runs/x
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "merging"))
import merging as m  # noqa: E402
import prune_sweep as ps  # noqa: E402

LIMIT_TOTAL, LIMIT_ACTIVE = 300_000_000, 150_000_000
EMBEDDINGS = ("transformer.wte.weight", "transformer.wpe.weight")


def sizes(header):
    cfg = m.gpt.config_from_header(header)
    C, L, H, V, P = cfg.n_embd, cfg.n_layer, 4 * cfg.n_embd, m.gpt.VOCAB, cfg.block_size
    return {"C": C, "L": L, "H": H, "V": V, "P": P,
            "emb": V * C + P * C,                        # wte (also the LM head) and wpe
            "attn_ln": L * (4 * C * C + 4 * C + 4 * C),  # attention weights and biases, ln_1 and ln_2
            "ffn": L * (2 * C * H + H + C), "neuron": 2 * C + 1, "ln_f": 2 * C}


def budget_a(s, rank):
    """Kept FFN neurons per expert (all layers) for design A, and its total and active parameters."""
    correction = rank * (s["V"] + s["C"]) + rank * (s["P"] + s["C"])
    fixed = s["attn_ln"] + s["ln_f"] + s["L"] * s["C"] + correction  # c_proj biases stay whole
    kept = min((LIMIT_TOTAL - s["emb"] - 4 * fixed) // (4 * s["neuron"]), s["L"] * s["H"])
    total = s["emb"] + 4 * (fixed + kept * s["neuron"])
    active = s["V"] * s["C"] + rank * (s["V"] + 2 * s["C"]) + s["L"] * 4 * s["C"] ** 2 + kept * 2 * s["C"]
    return kept, total, active


def budget_b(s):
    total = s["emb"] + s["attn_ln"] + 4 * (s["ffn"] + s["ln_f"])
    active = s["V"] * s["C"] + s["L"] * (4 * s["C"] ** 2 + 2 * s["C"] * s["H"])
    return total, active


_SVD = {}


def low_rank_toward(shared, own, rank, key):
    """shared + the rank-r truncated SVD of (own - shared); the SVD is cached under `key`."""
    if rank == 0:
        return shared.clone()
    if key not in _SVD:
        U, S, Vh = torch.linalg.svd((own - shared).float(), full_matrices=False)
        _SVD[key] = (U * S, Vh)
    US, Vh = _SVD[key]
    return (shared.float() + US[:, :rank] @ Vh[:rank]).to(shared.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--learned", default="", help="a learned-lambda model.bin, used as a shared source")
    parser.add_argument("--sources", default="generalist,learned,seed,expert-avg",
                        help="where the shared embeddings come from")
    parser.add_argument("--ranks", default="0,16,64", help="per-expert embedding correction ranks for design A")
    parser.add_argument("--b-sources", default="generalist,learned", help="design B shared-part sources")
    parser.add_argument("--calib-windows", type=int, default=128)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    dev = args.device
    header, gen = m.load_checkpoint("generalist", dev)
    _, seed = m.load_checkpoint("seed", dev)
    experts = {d: m.load_checkpoint(f"expert_{d}", dev)[1] for d in m.DOMAINS}
    sources = {"generalist": gen, "seed": seed,
               "expert-avg": {k: sum(experts[d][k] for d in m.DOMAINS) / 4 for k in EMBEDDINGS}}
    if args.learned and os.path.exists(args.learned):
        sources["learned"] = {k: v.to(dev) for k, v in m.gpt.read_checkpoint(args.learned)[1].items()}
    else:
        log(f"no learned mix at {args.learned!r}: skipping that source")
    model = m.new_model(header, dev)
    s = sizes(header)
    L, H = s["L"], s["H"]
    ranks = [int(r) for r in args.ranks.split(",") if r]
    budgets = {r: budget_a(s, r) for r in ranks}
    for r, (kept, total, active) in budgets.items():
        log(f"design A, embedding correction rank {r}: {kept:,} FFN neurons per expert "
            f"({kept / (L * H):.1%}), total {total / 1e6:.2f}M, active {active / 1e6:.2f}M")
        assert total <= LIMIT_TOTAL and active <= LIMIT_ACTIVE
    total_b, active_b = budget_b(s)
    log(f"design B: total {total_b / 1e6:.2f}M, active {active_b / 1e6:.2f}M")
    assert total_b <= LIMIT_TOTAL and active_b <= LIMIT_ACTIVE

    # per expert: its pruned FFN at each budget, found and repaired on its own text with its own parameters
    pruned = {}
    for d in m.DOMAINS:
        stats = ps.calibration_stats(model, experts[d], d, args.calib_windows, args.batch_size, args.seed, dev)
        order = ps.scores("taylor", experts[d], stats, L, args.seed)
        for r, (kept, _, _) in budgets.items():
            masks = ps.keep_masks(order, kept / (L * H), "global")
            pruned[d, r] = ps.pruned_params(experts[d], masks, "lstsq", stats)
            log(f"{d}: rank {r}: kept {sum(int(k.sum()) for k in masks):,} neurons")
        del stats
        torch.cuda.empty_cache()

    rows = {}

    def evaluate(name, route, step):
        """route(d) -> params used for domain d."""
        row = {d: m.score(model, route(d), args.val_windows, domains=[d])[d] for d in m.DOMAINS}
        row["mean"] = sum(row[d] for d in m.DOMAINS) / len(m.DOMAINS)
        rows[name] = row
        m.metrics.log(step=step, config=name, **{f"val_{k}": v for k, v in row.items()})
        log(f"{name:<52}" + " ".join(f"{row[k]:.4f}" for k in [*m.DOMAINS, "mean"]))

    def with_embeddings(params, name, d, rank):
        out = dict(params)
        for k in EMBEDDINGS:
            out[k] = low_rank_toward(sources[name][k], experts[d][k], rank, (name, d, k))
        return out

    def design_b(src):
        def shared(k):
            return k in EMBEDDINGS or (k.startswith("transformer.h.") and ".mlp." not in k)
        return lambda d: {k: (v if shared(k) else experts[d][k]) for k, v in src.items()}

    log(f"{'pool-val, each domain routed to its own route':<52}" + " ".join(f"{d:>6}" for d in [*m.DOMAINS, "mean"]))
    step = 0
    evaluate("reference: generalist", lambda d: gen, step); step += 1
    if "learned" in sources:
        evaluate("reference: learned mix", lambda d: sources["learned"], step); step += 1
    evaluate("ceiling: each domain's own expert", lambda d: experts[d], step); step += 1
    for r in ranks:
        evaluate(f"ceiling: own expert, pruned to the rank-{r} budget", lambda d, r=r: pruned[d, r], step); step += 1
    for name in [x for x in args.sources.split(",") if x in sources]:
        evaluate(f"A unpruned (over 300M): {name} embeddings",
                 lambda d, n=name: with_embeddings(experts[d], n, d, 0), step); step += 1
        for r in ranks:
            evaluate(f"A: {name} embeddings, correction rank {r}",
                     lambda d, n=name, r=r: with_embeddings(pruned[d, r], n, d, r), step); step += 1
    for name in [x for x in args.b_sources.split(",") if x in sources]:
        if not all(k in sources[name] for k in gen):
            continue
        evaluate(f"B (Method 2): {name} embeddings, attention, LNs", design_b(sources[name]), step); step += 1
    log("json " + json.dumps({"rows": rows, "budgets": {str(r): b for r, b in budgets.items()},
                              "budget_b": [total_b, active_b]}))


if __name__ == "__main__":
    main()
