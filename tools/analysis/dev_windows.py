"""
Error analysis over the committed per-window dev losses (results/*/window_losses.npz): where the
generalist-to-expert gap lives, how much each expert loses off its domain, and how the merges change
individual windows. Runs on a laptop; reads results/ only. Numbers feed error_analysis.md.

    python3 tools/analysis/dev_windows.py
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lib.common import DOMAINS  # noqa: E402


def load():
    """variant -> {domain: per-window loss}, for runs scored on dev."""
    runs = {}
    for path in glob.glob(os.path.join(ROOT, "results", "2026*", "window_losses.npz")):
        variant = json.load(open(path.replace("window_losses.npz", "run.json")))["variant"]
        runs[variant] = dict(np.load(path))
    return runs


def main():
    runs = load()
    gen, seed = runs["generalist"], runs["seed"]
    own = {d: runs[f"expert-{d}"][d] for d in DOMAINS}

    print("Where the gap lives: generalist minus own expert, per window")
    for d in DOMAINS:
        g = gen[d] - own[d]
        top = np.sort(g)[-len(g) // 10:].sum() / g.sum()
        print(f"  {d:<9} mean {g.mean():.3f} | expert better on {np.mean(g > 0):.0%} of windows | "
              f"median {np.median(g):.3f} | top 10% of windows hold {top:.0%} of the gap")

    print("Off-domain loss of each expert, next to the seed and the generalist")
    for e in DOMAINS:
        print(f"  {e}-expert: " + "; ".join(f"{d} {runs[f'expert-{e}'][d].mean():.3f} (seed {seed[d].mean():.3f}, "
                                            f"gen {gen[d].mean():.3f})" for d in DOMAINS if d != e))

    print("Share of each expert's own-domain gain over the seed that the generalist already has")
    for d in DOMAINS:
        s, g, x = seed[d].mean(), gen[d].mean(), own[d].mean()
        print(f"  {d:<9} seed {s:.3f} -> generalist {g:.3f} -> expert {x:.3f}: {(s - g) / (s - x):.0%}")

    if "learned-lambda" in runs:
        print("learned-lambda vs the generalist, per window")
        for d in DOMAINS:
            gain, adv = gen[d] - runs["learned-lambda"][d], gen[d] - own[d]
            print(f"  {d:<9} mean gain {gain.mean():+.4f} | windows improved {np.mean(gain > 0):.0%} | "
                  f"corr with expert advantage {np.corrcoef(gain, adv)[0, 1]:+.2f}")

    print("Per-domain change vs the generalist for every scored merge")
    for v in sorted(runs):
        if v not in ("generalist", "seed") and not v.startswith("expert-"):
            print(f"  {v:<16}" + " ".join(f"{d} {runs[v][d].mean() - gen[d].mean():+.3f}" for d in DOMAINS))


if __name__ == "__main__":
    main()
