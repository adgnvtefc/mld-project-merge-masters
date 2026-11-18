"""
Shared pieces of the merging scripts: load the provided checkpoints as task vectors relative to the seed,
put a merged state into a GPT module, score it on pool-val, and save a solution directory.

Task vectors: tau_i = theta_i - theta_seed for each SOURCE (the four experts and the generalist). Parameters
are dicts {name: float32 tensor} in the released checkpoint format (gpt.read_checkpoint), so the token
embedding keeps its padded rows until it is written back.
"""

import os
import shutil
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "code")]
import gpt  # noqa: E402
from lib import data, metrics  # noqa: E402,F401  (re-exported for the scripts)

DOMAINS = ["math", "code", "medicine", "history"]
EXPERTS = [f"expert_{d}" for d in DOMAINS]
SOURCES = EXPERTS + ["generalist"]


class Logger:
    """Prints and appends to <out>/train.log (copied into results/ by the launcher)."""

    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.file = open(os.path.join(out, "train.log"), "w")
        self.start = time.time()

    def __call__(self, message):
        line = f"[{time.time() - self.start:7.1f}s] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def load_checkpoint(name, device):
    header, params = gpt.read_checkpoint(os.path.join(ROOT, "models", f"{name}.bin"))
    return header, {k: v.to(device) for k, v in params.items()}


def load_task_vectors(sources, device):
    """(header, seed params, {source: task vector}); the header is the seed's, which keeps the muP fields."""
    header, seed = load_checkpoint("seed", device)
    taus = {}
    for name in sources:
        _, params = load_checkpoint(name, device)
        taus[name] = {k: params[k] - seed[k] for k in seed}
        del params
    return header, seed, taus


def combine(seed, terms):
    """seed + sum of coef * tau over terms [(coef, tau)], tensor by tensor."""
    return {k: seed[k] + sum(c * tau[k] for c, tau in terms) for k in seed}


def new_model(header, device):
    return gpt.GPT(gpt.config_from_header(header)).to(device).eval()


@torch.no_grad()
def set_params(model, params):
    """Copy checkpoint-format params into the module (the embedding drops its padded rows)."""
    for name, p in model.named_parameters():
        p.copy_(params[name][:p.shape[0]])
    return model


def score(model, params, windows):
    return data.val_loss(set_params(model, params), windows=windows, device=next(model.parameters()).device.type)


def save_solution(out, header, params, log):
    gpt.write_checkpoint(os.path.join(out, "model.bin"), header, params)
    shutil.copy(os.path.join(ROOT, "solutions", "gpt_model_def.py"), os.path.join(out, "model_def.py"))
    log(f"saved {out}/model.bin and model_def.py")


def fmt(losses):
    return " | ".join(f"{d} {losses[d]:.4f}" for d in DOMAINS) + f" | mean {losses['mean']:.4f}"


def floats(text):
    return [float(x) for x in text.split(",") if x.strip()]
