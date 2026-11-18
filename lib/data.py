"""
The pool, split the same way for every method. The last VAL_TOKENS of each pool/<domain>.bin are pool-val:
use it to tune, select and early-stop (dev is for comparing only). The rest is train.

    from lib import data
    tokens = data.pool("math", "train")                          # numpy uint16
    x, y = data.batch(["math", "code"], 2048, rng, "train")      # one random window per listed domain
    losses = data.val_loss(model, windows=64)                    # {"math": ..., ..., "mean": ...}

pool-val windows are the first `windows` non-overlapping 2,048-token windows of each domain's pool-val
slice, so every method and every run scores the same tokens.
"""

import functools
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.common import DOMAINS, ROOT

VAL_TOKENS = 2_000_000  # per domain, about 3% of the 60M pool
T = 2048
VOCAB = 50257


@functools.lru_cache(maxsize=None)
def _tokens(domain):
    header = np.fromfile(os.path.join(ROOT, "pool", f"{domain}.bin"), dtype=np.int32, count=256)
    tokens = np.memmap(os.path.join(ROOT, "pool", f"{domain}.bin"), dtype=np.uint16, mode="r", offset=1024)
    assert header[0] == 20240520 and tokens.size == header[2], f"pool/{domain}.bin: not a token file"
    return tokens


def pool(domain, split):
    """The train or val part of pool/<domain>.bin (a read-only memmap)."""
    tokens = _tokens(domain)
    if split == "train":
        return tokens[:-VAL_TOKENS]
    if split == "val":
        return tokens[-VAL_TOKENS:]
    raise ValueError(f"split must be 'train' or 'val', not {split!r}")


def batch(domains, seq_len, rng, split="train", device="cuda"):
    """One random (seq_len + 1)-token window from each listed domain: inputs and next-token targets."""
    rows = []
    for d in domains:
        tokens = pool(d, split)
        start = int(rng.integers(len(tokens) - seq_len - 1))
        rows.append(np.asarray(tokens[start:start + seq_len + 1], dtype=np.int64))
    xy = torch.from_numpy(np.stack(rows)).to(device)
    return xy[:, :-1], xy[:, 1:]


def val_windows(domain, windows):
    """`windows` non-overlapping T-token windows spread evenly over the domain's pool-val, as [n, T + 1] int64.

    Spread, not the first n in a row: pool-val's first 131K math tokens are 79% one document (a GMAT reading
    guide), which made the math expert look no better than the generalist on pool-val (fixed 2026-11-18)."""
    tokens = pool(domain, "val")
    n = min(windows, (len(tokens) - 1) // T)
    starts = np.linspace(0, len(tokens) - T - 1, n).astype(np.int64) if n > 1 else np.zeros(1, dtype=np.int64)
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + T + 1], dtype=np.int64) for s in starts]))


@torch.no_grad()
def val_loss(model, windows=64, batch_size=8, device="cuda", domains=DOMAINS):
    """Mean next-token cross-entropy on pool-val per domain, and their mean (bf16, like the official score)."""
    out = {}
    for d in domains:
        xy = val_windows(d, windows).to(device)
        total = 0.0
        for i in range(0, len(xy), batch_size):
            x, y = xy[i:i + batch_size, :-1], xy[i:i + batch_size, 1:]
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x)
            total += F.cross_entropy(logits[..., :VOCAB].float().reshape(-1, VOCAB), y.reshape(-1),
                                     reduction="sum").item()
        out[d] = total / (len(xy) * T)
    out["mean"] = sum(out[d] for d in domains) / len(domains)
    return out
