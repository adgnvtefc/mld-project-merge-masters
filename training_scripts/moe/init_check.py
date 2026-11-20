"""
MoE before any training. For each source of the shared parts (embeddings, attention, LayerNorms), with and
without per-expert embedding maps, build the MoE from the four experts' MLPs and score it on pool-val:

    oracle   every window goes to its own domain's expert (needs the label: an upper bound, not a model)
    router   every token is routed by the prefix router (what a submitted model would do)

The embedding maps are fitted with each token weighted by how often it occurs in that expert's pool train
data, and with the position embeddings included (moe_model.fit_embedding_maps). Shared-part sources: generalist, mean5 (mean of the four experts and the generalist), mean4 (mean of the four
experts, as in Branch-Train-MiX), and learned (the learned-lambda merge; needs --lambda-settings). References
on the same windows: the generalist alone, and each expert on its own domain (the 2.78-style target).

The best configuration by router mean is saved as a solution, so ./lab scores it on dev. Settings and every
table go to moe_settings.json and train.log. dev is never read here.

    python3 training_scripts/moe/init_check.py --out runs/x [--lambda-settings path/merge_settings.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "code"), os.path.dirname(os.path.abspath(__file__))]
import gpt  # noqa: E402
import moe_model as mm  # noqa: E402
from lib import data, metrics  # noqa: E402

DOMAINS = ["math", "code", "medicine", "history"]
EXPERTS = [f"expert_{d}" for d in DOMAINS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shared", default="generalist,mean5,mean4", help="sources of the shared parts")
    parser.add_argument("--maps", default="0,1", help="0: no embedding maps, 1: fitted maps")
    parser.add_argument("--lambda-settings", default="", help="merge_settings.json of a learned-lambda run")
    parser.add_argument("--router-windows", type=int, default=32, help="pool train windows per domain to fit the router")
    parser.add_argument("--val-windows", type=int, default=64, help="pool-val windows per domain")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class Logger:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.file, self.start = open(os.path.join(out, "train.log"), "w"), time.time()

    def __call__(self, message):
        line = f"[{time.time() - self.start:7.1f}s] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def load(name):
    return gpt.read_checkpoint(os.path.join(ROOT, "models", f"{name}.bin"))


def shared_source(name, header, params, args):
    """Checkpoint-format params for the shared parts."""
    if name == "generalist":
        return params["generalist"]
    if name in ("mean5", "mean4"):
        sources = EXPERTS + (["generalist"] if name == "mean5" else [])
        return {k: sum(params[s][k] for s in sources) / len(sources) for k in params["generalist"]}
    if name == "learned":
        table = json.load(open(args.lambda_settings))["lambda"]  # {source: {group: lam}}
        seed = params["seed"]

        def group_of(k):  # as in training_scripts/merging/learned_lambda.py
            parts = k.split(".")
            return ".".join(parts[1:-1]) if parts[1] == "h" else parts[1]
        return {k: seed[k] + sum(table[s][group_of(k)] * (params[s][k] - seed[k]) for s in table) for k in seed}
    raise SystemExit(f"unknown shared source {name!r}")


@torch.no_grad()
def domain_loss(model, windows, route=None, batch=8):
    """Mean next-token cross-entropy over [n, T + 1] windows (bf16, like the official score)."""
    total, device = 0.0, next(model.parameters()).device
    for i in range(0, len(windows), batch):
        xy = windows[i:i + batch].to(device)
        x, y = xy[:, :-1], xy[:, 1:]
        kwargs = {} if route is None else {"route": torch.full((len(x),), route, device=device)}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(x, **kwargs)
        total += F.cross_entropy(logits[..., :data.VOCAB].float().reshape(-1, data.VOCAB), y.reshape(-1),
                                 reduction="sum").item()
    return total / (len(windows) * data.T)


def fmt(losses):
    return " | ".join(f"{d} {losses[d]:.4f}" for d in DOMAINS) + f" | mean {losses['mean']:.4f}"


def with_mean(losses):
    return {**losses, "mean": sum(losses[d] for d in DOMAINS) / len(DOMAINS)}


def main():
    args = parse_args()
    log = Logger(args.out)
    log(f"settings: {json.dumps(vars(args))}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    shared_names = [s for s in args.shared.split(",") if s] + (["learned"] if args.lambda_settings else [])
    names = EXPERTS + ["generalist"] + (["seed"] if "learned" in shared_names else [])
    params = {name: load(name)[1] for name in names}
    header = gpt.read_header(os.path.join(ROOT, "models", "seed.bin"))  # keeps the muP fields, as in merging/

    val = {d: data.val_windows(d, args.val_windows) for d in DOMAINS}
    # How often each token occurs in each domain's pool train data: the weights of the embedding-map fit.
    counts = [np.bincount(data.pool(d, "train"), minlength=data.VOCAB)[:data.VOCAB] for d in DOMAINS]
    fit = []
    for d in DOMAINS:
        rows = [data.batch([d], data.T, rng, "train", device="cpu") for _ in range(args.router_windows)]
        fit.append(torch.cat([x for x, _ in rows]))

    # ---- references on the same windows ------------------------------------------------------------------
    refs = {}
    for name in ["generalist"] + EXPERTS:
        model = gpt.GPT(gpt.config_from_header(header)).to(args.device).eval()
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.copy_(params[name][k][:p.shape[0]])
        refs[name] = with_mean({d: domain_loss(model_wrapper(model), val[d], batch=args.batch) for d in DOMAINS})
        del model
    own = with_mean({d: refs[f"expert_{d}"][d] for d in DOMAINS})
    log(f"generalist            {fmt(refs['generalist'])}")
    log(f"own expert per domain {fmt(own)}")

    # ---- every configuration -----------------------------------------------------------------------------
    rows, best = [], None
    step = 0
    for shared_name in shared_names:
        shared = shared_source(shared_name, header, params, args)
        for use_maps in [bool(int(v)) for v in args.maps.split(",")]:
            model = mm.build(header, shared, [params[e] for e in EXPERTS], embed_map=use_maps, device=args.device)
            residuals = mm.fit_embedding_maps(
                model, [params[e]["transformer.wte.weight"] for e in EXPERTS],
                [params[e]["transformer.wpe.weight"] for e in EXPERTS], token_counts=counts) if use_maps else None
            mm.fit_prefix_router(model, fit, seed=args.seed)
            model.eval()
            accuracy = mm.router_accuracy(model, [val[d][:, :-1] for d in DOMAINS])
            oracle = with_mean({d: domain_loss(model, val[d], route=e, batch=args.batch) for e, d in enumerate(DOMAINS)})
            routed = with_mean({d: domain_loss(model, val[d], batch=args.batch) for d in DOMAINS})
            row = {"shared": shared_name, "maps": use_maps, "oracle": oracle, "router": routed,
                   "router_accuracy": accuracy, "map_residuals": residuals}
            rows.append(row)
            tag = f"{shared_name:<10} maps={int(use_maps)}"
            log(f"{tag} oracle {fmt(oracle)}")
            log(f"{tag} router {fmt(routed)}")
            log(f"{tag} router accuracy by position: "
                + ", ".join(f"{p}: {a:.3f}" for p, a in accuracy.items()))
            if residuals:
                log(f"{tag} map relative residuals: " + ", ".join(f"{d} {r:.3f}" for d, r in zip(DOMAINS, residuals)))
            metrics.log(step=step, oracle_mean=oracle["mean"], router_mean=routed["mean"],
                        router_accuracy=accuracy["all"], **{f"router_{d}": routed[d] for d in DOMAINS})
            step += 1
            if best is None or routed["mean"] < best[0]:
                if best is not None:
                    del best[1]
                best = [routed["mean"], model, row]
            else:
                del model
            torch.cuda.empty_cache() if args.device == "cuda" else None

    _, model, row = best
    log(f"best by router mean on pool-val: shared={row['shared']} maps={int(row['maps'])} | {fmt(row['router'])}")
    mm.save_solution(model, args.out)
    with open(os.path.join(args.out, "moe_settings.json"), "w") as f:
        json.dump({"chosen": {"shared": row["shared"], "maps": row["maps"]}, "references": {**refs, "own": own},
                   "rows": rows, "settings": vars(args)}, f, indent=2, default=str)
    log(f"saved {args.out}/model.pt, model_def.py and moe_settings.json")
    log("json " + json.dumps({"references": {"generalist": refs["generalist"], "own": own},
                              "rows": [{k: r[k] for k in ("shared", "maps", "oracle", "router")} for r in rows]}))


def model_wrapper(model):
    """gpt.GPT takes no route; give it the MoE call signature for domain_loss."""
    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, x, route=None):
            return self.inner(x)
    return Wrapped()


if __name__ == "__main__":
    main()
