"""
Diagnostics for why weight-space merges of the provided models fail. No model is saved; launch with
--no-eval. Everything is scored on pool-val (lib/data.py), never on dev.

    --which paths   loss along the straight line (1 - a) * A + a * B, a = 0, 0.1, ..., 1, for the generalist to
                    each expert and for two expert pairs. A bump above the straight line between the two ends
                    (a "barrier") means the models sit in different valleys of the loss, where averaging fails.
    --which scale   for merged models and their sources: the loss at the best single output temperature (how
                    much is plain miscalibration) and the size (RMS) of the hidden state after every block.
    --which tokens  each expert's per-token gain over the generalist on its own domain: how concentrated it is
                    in a few token types, how it splits by token frequency, and by position in the window.
    --which swaps   each expert's gain when one part (embeddings, LayerNorms, attention, MLP) of the generalist
                    is replaced by the expert's, and when that part of the expert is replaced by the generalist's.
    --which slices  the generalist's and each expert's own-domain loss at 10 evenly spaced places in each pool
                    file (and on pool-val), to check whether the pool files are ordered so that the held-out
                    tail looks different from the rest.

    python3 training_scripts/analysis/diagnostics.py --which paths --out runs/x
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "merging"))
import merging as m  # noqa: E402

VOCAB = 50257
PARTS = {
    "embeddings": lambda k: k in ("transformer.wte.weight", "transformer.wpe.weight"),
    "layernorms": lambda k: ".ln_" in k,
    "attention": lambda k: ".attn." in k,
    "mlp": lambda k: ".mlp." in k,
}


def table(log, title, rows, cols, fmt="{:>10.4f}"):
    log(title)
    log(f"{'':<30}" + "".join(f"{c:>10}" for c in cols))
    for name, values in rows.items():
        log(f"{name:<30}" + "".join(fmt.format(values[c]) if isinstance(values.get(c), float) else f"{'-':>10}"
                                    for c in cols))


def paths(args, log):
    _, gen = m.load_checkpoint("generalist", args.device)
    experts = {d: m.load_checkpoint(f"expert_{d}", args.device)[1] for d in m.DOMAINS}
    header = m.gpt.read_header(os.path.join(m.ROOT, "models", "seed.bin"))
    model = m.new_model(header, args.device)
    pairs = [("generalist", d, gen, experts[d]) for d in m.DOMAINS]
    pairs += [("math", "code", experts["math"], experts["code"]), ("medicine", "history", experts["medicine"],
                                                                     experts["history"])]
    alphas = [i / 10 for i in range(11)]
    out = {}
    for step, (a_name, b_name, a, b) in enumerate(pairs):
        name = f"{a_name} -> {b_name}"
        curve = []
        for alpha in alphas:
            params = {k: (1 - alpha) * a[k] + alpha * b[k] for k in a}
            curve.append(m.score(model, params, args.val_windows))
            m.metrics.log(step=step, path=name, alpha=alpha, **{f"val_{k}": v for k, v in curve[-1].items()})
        rows = {f"a = {alpha:.1f}": c for alpha, c in zip(alphas, curve)}
        table(log, f"path {name} (pool-val loss)", rows, [*m.DOMAINS, "mean"])
        barrier = {}
        for col in [*m.DOMAINS, "mean"]:
            ys = np.array([c[col] for c in curve])
            line = ys[0] + (ys[-1] - ys[0]) * np.array(alphas)
            barrier[col] = float((ys - line).max())
        log(f"barrier (largest bump above the straight line) {name}: "
            + " | ".join(f"{k} {v:+.4f}" for k, v in barrier.items()))
        out[name] = {"alphas": alphas, "curve": curve, "barrier": barrier}
    return out


def merged_models(args):
    """name -> params, for the scale diagnostic."""
    header, seed, taus = m.load_task_vectors(m.SOURCES, args.device)
    experts_sum = {k: sum(taus[e][k] for e in m.EXPERTS) for k in seed}
    models = {
        "generalist": m.combine(seed, [(1.0, taus["generalist"])]),
        "expert_math": m.combine(seed, [(1.0, taus["expert_math"])]),
        "average of experts": m.combine(seed, [(0.25, experts_sum)]),
        "ta lam_e 0.1 lam_g 1.0": m.combine(seed, [(0.1, experts_sum), (1.0, taus["generalist"])]),
    }
    if args.learned and os.path.exists(args.learned):
        models["learned-lambda"] = m.gpt.read_checkpoint(args.learned)[1]
        models["learned-lambda"] = {k: v.to(args.device) for k, v in models["learned-lambda"].items()}
    return header, models


@torch.no_grad()
def scale(args, log):
    header, models = merged_models(args)
    model = m.new_model(header, args.device)
    temps = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    rms = {}
    hooks = [block.register_forward_hook(lambda _m, _i, o, i=i: rms.setdefault(i, []).append(
        o.float().pow(2).mean(-1).sqrt().mean().item())) for i, block in enumerate(model.transformer.h)]
    out = {}
    for name, params in models.items():
        m.set_params(model, params)
        rms.clear()
        losses = {t: 0.0 for t in temps}
        n = 0
        for d in m.DOMAINS:
            xy = m.data.val_windows(d, args.val_windows).to(args.device)
            for i in range(0, len(xy), 4):
                x, y = xy[i:i + 4, :-1], xy[i:i + 4, 1:]
                with torch.autocast(args.device, dtype=torch.bfloat16):
                    logits = model(x)
                logits = logits[..., :VOCAB].float().reshape(-1, VOCAB)
                for t in temps:
                    losses[t] += F.cross_entropy(logits / t, y.reshape(-1), reduction="sum").item()
                n += y.numel()
        losses = {t: v / n for t, v in losses.items()}
        best_t = min(losses, key=losses.get)
        layer_rms = [float(np.mean(rms[i])) for i in sorted(rms)]
        out[name] = {"loss_t1": losses[1.0], "best_t": best_t, "loss_best_t": losses[best_t], "rms": layer_rms,
                     "by_temperature": {str(t): v for t, v in losses.items()}}
        m.metrics.log(step=len(out), model=name, loss_t1=losses[1.0], best_t=best_t, loss_best_t=losses[best_t])
        log(f"{name}: loss {losses[1.0]:.4f} at T=1; best T {best_t} -> {losses[best_t]:.4f} "
            f"(temperature recovers {losses[1.0] - losses[best_t]:.4f})")
    for h in hooks:
        h.remove()
    log("hidden-state RMS after each block (pool-val, all domains):")
    log(f"{'model':<26}" + "".join(f"{i:>7}" for i in range(len(out['generalist']['rms']))))
    for name, o in out.items():
        log(f"{name:<26}" + "".join(f"{v:>7.2f}" for v in o["rms"]))
    return out


@torch.no_grad()
def tokens(args, log):
    header, gen = m.load_checkpoint("generalist", args.device)
    model = m.new_model(header, args.device)
    out = {}
    for step, d in enumerate(m.DOMAINS):
        xy = m.data.val_windows(d, args.token_windows).to(args.device)
        x, y = xy[:, :-1], xy[:, 1:]

        def per_token(params):
            m.set_params(model, params)
            rows = []
            for i in range(0, len(x), 4):
                with torch.autocast(args.device, dtype=torch.bfloat16):
                    logits = model(x[i:i + 4])
                rows.append(F.cross_entropy(logits[..., :VOCAB].float().transpose(1, 2), y[i:i + 4],
                                            reduction="none"))
            return torch.cat(rows)

        g = per_token(gen)
        e = per_token(m.load_checkpoint(f"expert_{d}", args.device)[1])
        gain = (g - e).flatten().cpu().numpy()  # positive: the expert is better on that token
        target = y.flatten().cpu().numpy()
        by_type = np.bincount(target, weights=gain, minlength=VOCAB)
        positive = by_type[by_type > 0].sum()  # token types where the expert is better overall
        top = np.argsort(-by_type)
        counts_val = np.bincount(target, minlength=VOCAB)
        freq = np.bincount(np.asarray(m.data.pool(d, "train")[:20_000_000], dtype=np.int64), minlength=VOCAB)
        tok_freq = freq[target]
        edges = np.quantile(tok_freq, [0.2, 0.4, 0.6, 0.8])
        bucket = np.searchsorted(edges, tok_freq, side="right")
        positions = np.tile(np.arange(x.shape[1]), x.shape[0])
        pos_edges = [0, 16, 128, 512, 2048]
        res = {
            "mean_gain": float(gain.mean()),
            "tokens_where_expert_better": float((gain > 0).mean()),
            "share_of_positive_gain_top_types": {str(k): float(by_type[top[:k]].sum() / positive)
                                                 for k in (10, 100, 1000)},
            "types_where_generalist_better": float((by_type < 0).sum() / max((counts_val > 0).sum(), 1)),
            "distinct_types": int((counts_val > 0).sum()),
            "gain_by_frequency_quintile": [float(gain[bucket == q].mean()) for q in range(5)],
            "gain_by_position": {f"{a}-{b}": float(gain[(positions >= a) & (positions < b)].mean())
                                 for a, b in zip(pos_edges, pos_edges[1:])},
            "top_types": [{"id": int(t), "total_gain": float(by_type[t]), "count": int(counts_val[t]),
                           "mean_gain": float(by_type[t] / max(counts_val[t], 1))} for t in top[:25]],
        }
        out[d] = res
        m.metrics.log(step=step, domain=d, mean_gain=res["mean_gain"])
        log(f"{d}: mean per-token gain {res['mean_gain']:.4f} over {gain.size:,} tokens; expert better on "
            f"{res['tokens_where_expert_better']:.0%} of tokens")
        log(f"  share of the positive gain from the top 10 / 100 / 1000 token types (of {res['distinct_types']:,}): "
            + " / ".join(f"{v:.0%}" for v in res["share_of_positive_gain_top_types"].values())
            + f"; the generalist is better on {res['types_where_generalist_better']:.0%} of token types")
        log("  mean gain by token frequency quintile (rarest -> commonest): "
            + " ".join(f"{v:.3f}" for v in res["gain_by_frequency_quintile"]))
        log("  mean gain by position in the window: "
            + " | ".join(f"{k}: {v:.3f}" for k, v in res["gain_by_position"].items()))
    return out


def swaps(args, log):
    header, gen = m.load_checkpoint("generalist", args.device)
    model = m.new_model(header, args.device)
    base = m.score(model, gen, args.val_windows)
    out = {}
    for step, d in enumerate(m.DOMAINS):
        _, expert = m.load_checkpoint(f"expert_{d}", args.device)
        ref = m.score(model, expert, args.val_windows, domains=[d])[d]
        row = {"generalist": base[d], "expert": ref}
        for part, has in PARTS.items():
            into_gen = {k: expert[k] if has(k) else gen[k] for k in gen}
            into_exp = {k: gen[k] if has(k) else expert[k] for k in gen}
            row[f"generalist + expert {part}"] = m.score(model, into_gen, args.val_windows, domains=[d])[d]
            row[f"expert - own {part}"] = m.score(model, into_exp, args.val_windows, domains=[d])[d]
        out[d] = row
        m.metrics.log(step=step, domain=d, **{k.replace(" ", "_"): v for k, v in row.items()})
        del expert
    names = list(out[m.DOMAINS[0]])
    rows = {n: {d: out[d][n] for d in m.DOMAINS} for n in names}
    for n in names:
        rows[n]["mean"] = float(np.mean([out[d][n] for d in m.DOMAINS]))
    table(log, "own-domain pool-val loss when one part is swapped (expert - own X = expert with the "
               "generalist's X):", rows, [*m.DOMAINS, "mean"])
    gap = {d: out[d]["generalist"] - out[d]["expert"] for d in m.DOMAINS}
    log("share of the expert's gain recovered by each swap (1 = the expert, 0 = the generalist):")
    for n in names[2:]:
        log(f"  {n:<34}" + " ".join(f"{d} {(out[d]['generalist'] - out[d][n]) / gap[d]:+.2f}" for d in m.DOMAINS))
    return out


@torch.no_grad()
def slices(args, log):
    header, gen = m.load_checkpoint("generalist", args.device)
    model = m.new_model(header, args.device)
    T, n_win = 2048, args.val_windows
    out = {}
    for step, d in enumerate(m.DOMAINS):
        _, expert = m.load_checkpoint(f"expert_{d}", args.device)
        tokens = m.data._tokens(d)
        starts = [int(f * (len(tokens) - n_win * T - 1)) for f in np.linspace(0, 1, 10)]
        rows = []
        for i, start in enumerate(starts):
            xy = torch.from_numpy(np.stack([np.asarray(tokens[start + w * T:start + (w + 1) * T + 1], dtype=np.int64)
                                            for w in range(n_win)])).to(args.device)
            losses = []
            for params in (gen, expert):
                m.set_params(model, params)
                total = 0.0
                for j in range(0, n_win, 8):
                    with torch.autocast(args.device, dtype=torch.bfloat16):
                        logits = model(xy[j:j + 8, :-1])
                    total += F.cross_entropy(logits[..., :VOCAB].float().reshape(-1, VOCAB),
                                             xy[j:j + 8, 1:].reshape(-1), reduction="sum").item()
                losses.append(total / (n_win * T))
            rows.append({"position": start / len(tokens), "generalist": losses[0], "expert": losses[1],
                         "gap": losses[0] - losses[1]})
            m.metrics.log(step=step * 10 + i, domain=d, position=start / len(tokens), generalist=losses[0],
                          expert=losses[1])
        out[d] = rows
        log(f"{d}: own-domain loss at 10 places in pool/{d}.bin ({n_win} windows each); pool-val is the last "
            f"{m.data.VAL_TOKENS / len(tokens):.1%}")
        for r in rows:
            log(f"  at {r['position']:5.1%}: generalist {r['generalist']:.4f} expert {r['expert']:.4f} "
                f"gap {r['gap']:+.4f}")
        del expert
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--which", required=True, choices=["paths", "scale", "tokens", "swaps", "slices"])
    parser.add_argument("--val-windows", type=int, default=32, help="pool-val windows per domain")
    parser.add_argument("--token-windows", type=int, default=16, help="tokens: pool-val windows per domain")
    parser.add_argument("--learned", default="", help="scale: a learned-lambda model.bin to include")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    log = m.Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    result = {"paths": paths, "scale": scale, "tokens": tokens, "swaps": swaps, "slices": slices}[args.which](args, log)
    log("json " + json.dumps(result))


if __name__ == "__main__":
    main()
