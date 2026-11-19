"""
The MoE model: shared embeddings, attention and LayerNorms, and one MLP per expert in every block. Each
token goes through exactly one expert (top-1), so a token uses about as many weights as the provided GPT
(~124M active) while the model holds ~295M.

    model = MoEGPT(MoEConfig(n_experts=4))      # usually built by build() from the provided checkpoints
    logits = model(idx)                         # route each token with the prefix router
    logits = model(idx, route=domains)          # or force the route: [B] (one expert per sequence) or [B, T]

Routing. The prefix router predicts the domain of token t from the mean of the (shared) token embeddings
of tokens 0..t, so it only sees earlier tokens and the model stays causal. It is a linear classifier,
fitted in closed form by fit_prefix_router() and trainable after that. Every block uses the same route.

Dispatch, not masking. Tokens are sorted by expert and each expert's MLP runs only on its own tokens.
create_submission.py counts active parameters from matrix-multiply FLOPs, so running all experts and
masking would count as ~295M active, over the 150M limit.

Embedding maps (optional, MoEConfig.embed_map). Each expert's MLPs were trained with its own token
embeddings, and the swaps diagnostic showed they break with another model's. A per-expert 768x768 map M_e,
fitted so that shared_wte @ M_e ~ expert_wte (fit_embedding_maps), moves the input embedding into expert e's
basis, and M_e^T on the final hidden state does the same for the tied output layer
(logits = h @ W_e^T ~ (h @ M_e^T) @ W_shared^T). The fit weights tokens by how often they occur in the
expert's pool data and includes the position embeddings, since M_e is applied to their sum. Cost: 0.59M
parameters per expert and ~1.2M active.

This file is also the solution's model_def.py: save_solution() copies it into the output directory, and
create_submission.py calls load_model(). It needs only torch and code/gpt.py.
"""

import os
import shutil
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import gpt  # code/gpt.py; create_submission.py puts code/ on the import path


@dataclass
class MoEConfig(gpt.Config):
    n_experts: int = 4
    embed_map: bool = False


# ---- routing ------------------------------------------------------------------------------------------

class Dispatch:
    """One routing decision, reused by every block: sort tokens by expert, run each expert on its slice,
    put the results back in token order. route: [N] expert index per token."""

    def __init__(self, route, n_experts):
        self.order = torch.argsort(route, stable=True)
        self.counts = torch.bincount(route, minlength=n_experts).tolist()  # one host sync per forward
        self.inverse = torch.empty_like(self.order)
        self.inverse[self.order] = torch.arange(len(self.order), device=route.device)

    def __call__(self, fn, h):
        """fn(expert, rows) -> outputs for those rows; h [N, C]. Experts with no tokens never run."""
        pieces = h[self.order].split(self.counts)
        out = torch.cat([fn(e, p).to(h.dtype) for e, p in enumerate(pieces) if self.counts[e]])
        return out[self.inverse]


def prefix_features(embeddings):
    """Mean of the embeddings of tokens 0..t at every position t: [B, T, C] -> [B, T, C]. Causal."""
    T = embeddings.shape[1]
    steps = torch.arange(1, T + 1, device=embeddings.device, dtype=embeddings.dtype)
    return embeddings.cumsum(dim=1) / steps[None, :, None]


class PrefixRouter(nn.Module):
    def __init__(self, n_embd, n_experts):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_experts)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embeddings):
        """Router logits [B, T, n_experts] from the token embeddings (not the position embeddings)."""
        return self.proj(prefix_features(embeddings))


# ---- model --------------------------------------------------------------------------------------------

class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = gpt.Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.experts = nn.ModuleList(gpt.MLP(config) for _ in range(config.n_experts))

    def forward(self, x, dispatch):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x).reshape(-1, x.shape[-1])
        return x + dispatch(lambda e, rows: self.experts[e](rows), h).view_as(x)


class MoEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # gpt.optimizer_groups reads config.width_mult
        # Same names as gpt.GPT for the shared parts, so gpt.optimizer_groups sorts them the same way.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(MoEBlock(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # tied, as in the provided models
        self.router = PrefixRouter(config.n_embd, config.n_experts)
        if config.embed_map:
            eye = torch.eye(config.n_embd).repeat(config.n_experts, 1, 1)
            self.embed_maps = nn.Parameter(eye)  # [E, C, C]; identity until fitted

    def router_logits(self, idx):
        """[B, T, n_experts]. The router reads the embeddings without moving them (detached)."""
        return self.router(self.transformer.wte(idx).detach())

    def route_of(self, idx, route=None):
        """Expert index per token, [B, T]: the given route (per sequence or per token), else the router's."""
        B, T = idx.shape
        if route is None:
            return self.router_logits(idx).argmax(dim=-1)
        route = torch.as_tensor(route, device=idx.device, dtype=torch.long)
        return route[:, None].expand(B, T) if route.dim() == 1 else route

    def forward(self, idx, route=None):
        t, C = self.transformer, self.config.n_embd
        dispatch = Dispatch(self.route_of(idx, route).reshape(-1), self.config.n_experts)
        x = t.wte(idx) + t.wpe(torch.arange(idx.shape[1], device=idx.device))
        if self.config.embed_map:
            x = dispatch(lambda e, rows: rows @ self.embed_maps[e], x.reshape(-1, C)).view_as(x)
        for block in t.h:
            x = block(x, dispatch)
        h = t.ln_f(x) / self.config.width_mult
        if self.config.embed_map:
            h = dispatch(lambda e, rows: rows @ self.embed_maps[e].T, h.reshape(-1, C)).view_as(h)
        return self.lm_head(h)


# ---- building from the provided checkpoints -----------------------------------------------------------

SHARED_BLOCK = ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                "attn.c_proj.bias", "ln_2.weight", "ln_2.bias")
EXPERT_BLOCK = ("mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")


@torch.no_grad()
def build(header, shared, experts, embed_map=False, device="cpu"):
    """An MoEGPT from checkpoint-format dicts ({name: tensor}, as gpt.read_checkpoint returns): `shared` gives
    the embeddings, attention and LayerNorms, experts[e] gives expert e's MLPs. The router starts at zero
    (every token to expert 0) and the embedding maps at the identity: fit both before using the model."""
    config = MoEConfig(**asdict(gpt.config_from_header(header)), n_experts=len(experts), embed_map=embed_map)
    model = MoEGPT(config)
    t = model.transformer
    t.wte.weight.copy_(shared["transformer.wte.weight"][:config.vocab_size])
    t.wpe.weight.copy_(shared["transformer.wpe.weight"])
    t.ln_f.weight.copy_(shared["transformer.ln_f.weight"])
    t.ln_f.bias.copy_(shared["transformer.ln_f.bias"])
    for i, block in enumerate(t.h):
        for name in SHARED_BLOCK:
            block.get_parameter(name).copy_(shared[f"transformer.h.{i}.{name}"])
        for e, params in enumerate(experts):
            for name in EXPERT_BLOCK:
                block.experts[e].get_parameter(name.removeprefix("mlp.")).copy_(params[f"transformer.h.{i}.{name}"])
    return model.to(device)


@torch.no_grad()
def fit_embedding_maps(model, expert_wtes, expert_wpes, token_counts=None, ridge=1e-4):
    """Fit each expert's map M_e so the model's input embedding, moved by M_e, matches the expert's own input
    embedding, on average over the text the model will see; sets model.embed_maps.

    The model's input at a token t in position p is s = wte[t] + wpe[p]; expert e's would be
    u = wte_e[t] + wpe_e[p]. M_e minimizes E ||s M - u||^2 with t drawn from token_counts[e] (how often each
    token occurs in expert e's pool data; uniform if None) and p uniform over the positions, as in the
    2,048-token eval windows. With t and p independent the expectation has a closed form, cross terms
    included, so no data pass is needed:

        G = E[s^T s] = sum_t f_t wte[t]^T wte[t] + mean_p wpe[p]^T wpe[p] + mt^T mp + mp^T mt
        R = E[s^T u] = sum_t f_t wte[t]^T wte_e[t] + mean_p wpe[p]^T wpe_e[p] + mt^T mp_e + mp^T mt_e
        M = (G + ridge) ^-1 R

    (mt, mp: the mean token and position embedding of the shared model; mt_e, mp_e: the expert's.)
    Returns each expert's relative residual sqrt(E ||s M - u||^2 / E ||u||^2): 0 is a perfect linear match,
    1 is no better than mapping everything to zero."""
    assert model.config.embed_map, "build the model with embed_map=True"
    V, C = model.config.vocab_size, model.config.n_embd
    Ws = model.transformer.wte.weight.double()
    Ps = model.transformer.wpe.weight.double()
    device, eye = Ws.device, torch.eye(C, device=Ws.device, dtype=torch.float64)
    mp_s = Ps.mean(dim=0)
    residuals = []
    for e in range(model.config.n_experts):
        We = expert_wtes[e][:V].to(device).double()
        Pe = expert_wpes[e].to(device).double()
        f = torch.ones(V, dtype=torch.float64) if token_counts is None else torch.as_tensor(token_counts[e])[:V]
        f = f.to(device).double()
        f = f / f.sum()
        mt_s, mt_e, mp_e = f @ Ws, f @ We, Pe.mean(dim=0)
        Wf = Ws * f[:, None]
        G = Wf.T @ Ws + Ps.T @ Ps / len(Ps) + torch.outer(mt_s, mp_s) + torch.outer(mp_s, mt_s)
        R = Wf.T @ We + Ps.T @ Pe / len(Ps) + torch.outer(mt_s, mp_e) + torch.outer(mp_s, mt_e)
        U = (f * (We * We).sum(dim=1)).sum() + (Pe * Pe).sum(dim=1).mean() + 2 * mt_e @ mp_e  # E ||u||^2
        M = torch.linalg.solve(G + ridge * G.diagonal().mean() * eye, R)
        err = (M * (G @ M)).sum() - 2 * (M * R).sum() + U  # E ||s M - u||^2, expanded
        residuals.append((err.clamp(min=0) / U).sqrt().item())
        model.embed_maps[e].copy_(M.float())
    return residuals


@torch.no_grad()
def fit_prefix_router(model, windows, positions=128, ridge=1e-3, seed=0):
    """Fit the router in closed form: ridge regression from prefix features to the one-hot domain, then
    argmax. windows[e] is a [n, T] int64 tensor of expert e's domain text (pool train, never dev). Each
    window contributes `positions` positions, half log-spaced from the start (where routing is hardest) and
    half uniform."""
    T = windows[0].shape[1]
    half = positions // 2
    early = torch.unique(torch.logspace(0, torch.log10(torch.tensor(float(T))), half).long() - 1).clamp(0, T - 1)
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for e, idx in enumerate(windows):
        idx = idx.to(model.transformer.wte.weight.device)
        f = prefix_features(model.transformer.wte(idx).float())  # [n, T, C]
        for row in range(f.shape[0]):
            pos = torch.cat([early, torch.randint(0, T, (positions - len(early),), generator=g)]).to(f.device)
            feats.append(f[row, pos])
            labels.append(torch.full((len(pos),), e, device=f.device))
    X = torch.cat(feats).double()
    X = torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], dim=1)  # bias column
    Y = F.one_hot(torch.cat(labels), model.config.n_experts).double()
    gram = X.T @ X
    gram += ridge * gram.diagonal().mean() * torch.eye(gram.shape[0], device=X.device, dtype=X.dtype)
    coef = torch.linalg.solve(gram, X.T @ Y)  # [C + 1, E]
    model.router.proj.weight.copy_(coef[:-1].T.float())
    model.router.proj.bias.copy_(coef[-1].float())


@torch.no_grad()
def router_accuracy(model, windows, at=(0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047)):
    """{position: share of windows routed to their own domain at that position}, plus "all": over every
    position. windows[e] as in fit_prefix_router (use pool-val windows here)."""
    hits, total, all_hits, all_total = {p: 0 for p in at}, 0, 0, 0
    for e, idx in enumerate(windows):
        idx = idx.to(model.transformer.wte.weight.device)
        right = model.router_logits(idx).argmax(dim=-1) == e  # [n, T]
        for p in at:
            if p < right.shape[1]:
                hits[p] += right[:, p].sum().item()
        total += right.shape[0]
        all_hits += right.sum().item()
        all_total += right.numel()
    return {**{p: hits[p] / total for p in at}, "all": all_hits / all_total}


# ---- solution directory -------------------------------------------------------------------------------

def save_solution(model, out):
    """Write model.pt and model_def.py (this file) into out."""
    os.makedirs(out, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, os.path.join(out, "model.pt"))
    shutil.copy(os.path.abspath(__file__), os.path.join(out, "model_def.py"))


def load_model(directory, device):
    saved = torch.load(os.path.join(directory, "model.pt"), map_location="cpu", weights_only=True)
    model = MoEGPT(MoEConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()
