"""
model_def.py for a routed design-A solution (training_scripts/routed/): shared embeddings, four experts with their
own attention, pruned FFNs and low-rank embedding corrections, and a causal prefix router with no parameters.

Files in the solution directory: model.pt (state_dict) and config.json (sizes, FFN widths, rank, commit point).
create_submission.py calls load_model(directory, device).

Routing, per sequence: for positions t < commit, all experts run and the output is the mixture of their
next-token distributions weighted by softmax(-loss of each expert on tokens 1..t) (uniform at t = 0). From position
`commit` on, the expert with the lowest loss on tokens 1..commit-1 predicts alone; it reruns the whole prefix,
since its attention needs its own states for earlier tokens. Every prediction depends only on earlier tokens.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import gpt  # code/gpt.py; create_submission.py puts code/ on the import path


class MLP(nn.Module):
    def __init__(self, n_embd, width):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, width)
        self.c_proj = nn.Linear(width, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = gpt.Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, width)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Expert(nn.Module):
    def __init__(self, config, widths, rank):
        super().__init__()
        self.h = nn.ModuleList(Block(config, w) for w in widths)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.rank = rank
        if rank:
            self.wte_u = nn.Parameter(torch.zeros(config.vocab_size, rank))
            self.wte_v = nn.Parameter(torch.zeros(rank, config.n_embd))
            self.wpe_u = nn.Parameter(torch.zeros(config.block_size, rank))
            self.wpe_v = nn.Parameter(torch.zeros(rank, config.n_embd))


class RoutedExperts(nn.Module):
    def __init__(self, config, widths, rank, commit):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.experts = nn.ModuleList(Expert(config, w, rank) for w in widths)
        self.commit = commit
        self.force = None  # an expert index: that expert predicts every token (oracle routing, for analysis)
        self.last_nll = self.last_choice = None

    def expert_logits(self, e, idx):
        ex, T = self.experts[e], idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        if ex.rank:
            x = x + F.embedding(idx, ex.wte_u) @ ex.wte_v + ex.wpe_u[:T] @ ex.wpe_v
        for block in ex.h:
            x = block(x)
        h = ex.ln_f(x) / self.config.width_mult
        logits = F.linear(h, self.wte.weight)
        if ex.rank:
            logits = logits + F.linear(F.linear(h, ex.wte_v), ex.wte_u)
        return logits

    def forward(self, idx, route=None):
        """route: optional [B] expert index per sequence (training and oracle scoring); None = the prefix router."""
        if self.force is not None:
            return self.expert_logits(self.force, idx)
        B, T = idx.shape
        if route is not None:
            out = None
            for e in range(len(self.experts)):
                rows = (route == e).nonzero().squeeze(1)
                if rows.numel():
                    logits = self.expert_logits(e, idx[rows])
                    if out is None:
                        out = logits.new_empty(B, T, logits.shape[-1])
                    out[rows] = logits
            return out
        E, k = len(self.experts), min(self.commit, T)
        logp = torch.stack([F.log_softmax(self.expert_logits(e, idx[:, :k]).float(), -1) for e in range(E)])
        tok_nll = torch.zeros(E, B, k, device=idx.device)
        if k > 1:
            target = idx[:, 1:k].expand(E, B, k - 1).unsqueeze(-1)
            tok_nll[:, :, 1:] = -logp[:, :, :-1].gather(-1, target).squeeze(-1)
        nll = tok_nll.cumsum(-1)  # nll[e, b, t]: expert e's loss on tokens 1..t, known at position t
        prefix = torch.logsumexp(logp + F.log_softmax(-nll, dim=0)[..., None], dim=0)  # [B, k, V]
        self.last_nll = nll
        if k == T:
            return prefix
        choice = nll[:, :, k - 1].argmin(0)
        self.last_choice = choice
        out = torch.empty(B, T, prefix.shape[-1], device=idx.device, dtype=torch.float32)
        out[:, :k] = prefix
        for e in range(E):
            rows = (choice == e).nonzero().squeeze(1)
            if rows.numel():
                out[rows, k:] = self.expert_logits(e, idx[rows])[:, k:].float()
        return out


def build(config_dict):
    config = gpt.Config(**config_dict["gpt_config"])
    return RoutedExperts(config, config_dict["widths"], config_dict["rank"], config_dict["commit"])


def load_model(directory, device):
    with open(os.path.join(directory, "config.json")) as f:
        config_dict = json.load(f)
    model = build(config_dict)
    state = torch.load(os.path.join(directory, "model.pt"), map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device).eval()
