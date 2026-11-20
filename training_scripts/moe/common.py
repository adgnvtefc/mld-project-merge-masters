"""
Shared pieces of the MoE scripts: load the provided checkpoints, make the shared parts, build the starting
MoE (shared parts + the experts' MLPs, embedding maps and router fitted in closed form), and score a model
on pool-val. Training scripts and the final recipe build their starting model with build_initial(), so it is
exactly the model init_check.py scored.

Everything here reads only models/ and pool/ (train for fitting, val for scoring); dev is never read.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (ROOT, os.path.join(ROOT, "code"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)
import gpt  # noqa: E402
import moe_model as mm  # noqa: E402
from lib import data  # noqa: E402

DOMAINS = ["math", "code", "medicine", "history"]
EXPERTS = [f"expert_{d}" for d in DOMAINS]


class Logger:
    """Prints and appends to <out>/train.log (copied into results/ by the launcher)."""

    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.file, self.start = open(os.path.join(out, "train.log"), "w"), time.time()

    def __call__(self, message):
        line = f"[{time.time() - self.start:7.1f}s] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def model_path(name):
    return os.path.join(ROOT, "models", f"{name}.bin")


def seed_header():
    """The seed's header: it keeps the muP fields, as in the merging scripts."""
    return gpt.read_header(model_path("seed"))


def load_params(names):
    """{name: checkpoint-format params} for the provided models, on the CPU."""
    return {name: gpt.read_checkpoint(model_path(name))[1] for name in names}


def sources_for(shared):
    """The checkpoints shared_source(shared) needs, besides the experts."""
    return EXPERTS + ["generalist"] + (["seed"] if shared == "learned" else [])


def shared_source(name, params, lambda_settings=""):
    """Checkpoint-format params for the shared parts (embeddings, attention, LayerNorms)."""
    if name == "generalist":
        return params["generalist"]
    if name in ("mean5", "mean4"):
        sources = EXPERTS + (["generalist"] if name == "mean5" else [])
        return {k: sum(params[s][k] for s in sources) / len(sources) for k in params["generalist"]}
    if name == "learned":
        if not lambda_settings:
            raise SystemExit("shared source 'learned' needs --lambda-settings (a learned-lambda merge_settings.json)")
        table = json.load(open(lambda_settings))["lambda"]  # {source: {group: lam}}
        seed = params["seed"]

        def group_of(k):  # as in training_scripts/merging/learned_lambda.py
            parts = k.split(".")
            return ".".join(parts[1:-1]) if parts[1] == "h" else parts[1]
        return {k: seed[k] + sum(table[s][group_of(k)] * (params[s][k] - seed[k]) for s in table) for k in seed}
    raise SystemExit(f"unknown shared source {name!r}")


def token_counts():
    """How often each token occurs in each domain's pool train data (the weights of the embedding-map fit)."""
    return [np.bincount(data.pool(d, "train"), minlength=data.VOCAB)[:data.VOCAB] for d in DOMAINS]


def router_windows(n, seed):
    """n random pool train windows per domain, [n, T] int64 each, for fitting the router. Its own RNG, so
    refitting never moves a training script's data stream."""
    rng = np.random.default_rng(seed + 10_000)
    return [torch.cat([data.batch([d], data.T, rng, "train", device="cpu")[0] for _ in range(n)]) for d in DOMAINS]


def fit_router(model, windows, seed):
    model.eval()
    mm.fit_prefix_router(model, windows, seed=seed)


def build_initial(shared, maps, device, params=None, lambda_settings="", counts=None, windows=None, seed=0):
    """The starting MoE: shared parts from `shared`, the four experts' MLPs, embedding maps (if `maps`) and
    the prefix router, both fitted in closed form. Returns (model, map residuals or None)."""
    params = params if params is not None else load_params(sources_for(shared))
    model = mm.build(seed_header(), shared_source(shared, params, lambda_settings),
                     [params[e] for e in EXPERTS], embed_map=maps, device=device)
    residuals = None
    if maps:
        residuals = mm.fit_embedding_maps(
            model, [params[e]["transformer.wte.weight"] for e in EXPERTS],
            [params[e]["transformer.wpe.weight"] for e in EXPERTS],
            token_counts=counts if counts is not None else token_counts())
    fit_router(model, windows if windows is not None else router_windows(32, seed), seed)
    return model, residuals


@torch.no_grad()
def domain_loss(model, windows, route=None, batch=8):
    """Mean next-token cross-entropy over [n, T + 1] windows (bf16, like the official score). route: an
    expert index to force for every window (MoE only), or None for the model's own routing."""
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


def with_mean(losses):
    return {**losses, "mean": sum(losses[d] for d in DOMAINS) / len(DOMAINS)}


def fmt(losses):
    return " | ".join(f"{d} {losses[d]:.4f}" for d in DOMAINS) + f" | mean {losses['mean']:.4f}"
