"""
CPU checks for moe_model.py, on a tiny config (2 layers, width 64) plus one full-size parameter count. Run
on a laptop before pushing; needs torch 2.10.0 and no data or GPU:

    python training_scripts/moe/test_moe.py

Checks: the MoE equals gpt.GPT when every expert is the same model; forced routing to expert e equals that
expert's own GPT when the shared parts come from it; per-token dispatch equals a dense masked reference;
causality and active parameters by create_submission.py's own functions; the full-size budget; save/load;
the closed-form router and embedding-map fits; gradients and optimizer groups.
"""

import os
import sys
import tempfile
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "code"), os.path.dirname(os.path.abspath(__file__))]
import create_submission as official  # noqa: E402
import gpt  # noqa: E402
import moe_model as mm  # noqa: E402

torch.manual_seed(0)
TINY = gpt.Config(block_size=2048, vocab_size=gpt.VOCAB, n_layer=2, n_head=4, n_embd=64, width_mult=2.0,
                  attn_mult=1.5)
E = 4
FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f'  ({detail})' if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def header_for(config):
    h = np.zeros(256, dtype=np.int32)
    h[:8] = [gpt.MODEL_MAGIC, 3, config.block_size, config.vocab_size, config.n_layer, config.n_head,
             config.n_embd, config.vocab_size + 47]  # padded embedding rows, as in the released files
    h[8], h[10] = gpt.MUP_MAGIC, 1
    h[11:13] = np.array([config.width_mult, config.attn_mult], dtype=np.float32).view(np.int32)
    return h


def random_gpt(config):
    model = gpt.GPT(config).eval()
    with torch.no_grad():
        for p in model.parameters():  # larger than default init, so differences between models show
            p.add_(torch.randn_like(p) * 0.05)
    return model


def params_of(model):
    """Checkpoint-format dict, with padded embedding rows like gpt.read_checkpoint."""
    out = {k: v.detach().clone() for k, v in model.named_parameters()}
    out["transformer.wte.weight"] = torch.cat([out["transformer.wte.weight"], torch.zeros(47, model.config.n_embd)])
    return out


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


header = header_for(TINY)
gpts = [random_gpt(TINY) for _ in range(E)]
experts = [params_of(g) for g in gpts]
idx = torch.randint(0, gpt.VOCAB, (2, 96))

# ---- 1. all experts the same model: the MoE is that model ----------------------------------------------
moe = mm.build(header, experts[0], [experts[0]] * E).eval()
route = torch.randint(0, E, (2, 96))
with torch.no_grad():
    d = max_diff(moe(idx, route=route), gpts[0](idx))
check("identical experts equal gpt.GPT (random per-token route)", d < 1e-4, f"max diff {d:.1e}")

# ---- 2. shared parts from expert 2, forced route 2: the MoE is expert 2 -----------------------------------
moe = mm.build(header, experts[2], experts).eval()
with torch.no_grad():
    d = max_diff(moe(idx, route=torch.tensor([2, 2])), gpts[2](idx))
check("forced route to expert e equals expert e", d < 1e-4, f"max diff {d:.1e}")

# ---- 3. per-token dispatch equals a dense masked reference -----------------------------------------------


@torch.no_grad()
def dense_reference(model, idx, route):
    t = model.transformer
    x = t.wte(idx) + t.wpe(torch.arange(idx.shape[1]))
    for block in t.h:
        x = x + block.attn(block.ln_1(x))
        h = block.ln_2(x)
        x = x + sum((route == e).unsqueeze(-1) * block.experts[e](h) for e in range(E))
    return model.lm_head(t.ln_f(x) / model.config.width_mult)


with torch.no_grad():
    d = max_diff(moe(idx, route=route), dense_reference(moe, idx, route))
check("per-token dispatch equals dense masking", d < 1e-4, f"max diff {d:.1e}")
with torch.no_grad():
    d = max_diff(moe(idx, route=torch.tensor([1, 3])), dense_reference(moe, idx, torch.tensor([[1], [3]]).expand(2, 96)))
check("per-sequence route equals the same per-token route", d < 1e-4, f"max diff {d:.1e}")

# ---- 4. router fit on separable synthetic domains --------------------------------------------------------
# Each fake domain uses its own 20 token ids. (With thousands of random ids per domain, every domain's mean
# embedding is ~0 and no linear router could separate them; real domains differ in which tokens they use.)
windows = [torch.randint(e * 20, e * 20 + 20, (8, 2048)) for e in range(E)]
mm.fit_prefix_router(moe, windows)
acc = mm.router_accuracy(moe, [torch.randint(e * 20, e * 20 + 20, (4, 2048)) for e in range(E)])
check("prefix router separates synthetic domains", acc["all"] > 0.95 and acc[2047] == 1.0,
      f"all {acc['all']:.3f}, first token {acc[0]:.2f}, last {acc[2047]:.2f}")

# ---- 5. causality, with the fitted router (official check) -----------------------------------------------
window = torch.cat([torch.randint(0, 5000, (1024,)), torch.randint(10000, 15000, (1024,))])  # domain changes midway
check("causal (create_submission.causal)", official.causal(moe, window))
moe_maps = mm.build(header, experts[0], experts, embed_map=True).eval()
moe_maps.router.load_state_dict(moe.router.state_dict())
check("causal with embedding maps", official.causal(moe_maps, window))

# ---- 6. active parameters (official estimate) ------------------------------------------------------------
short = window[:256]
active_gpt = official.active_parameters(gpts[0], short, "cpu")
active_moe = official.active_parameters(moe, short, "cpu")
active_maps = official.active_parameters(moe_maps, short, "cpu")
router_cost = TINY.n_embd * E
map_cost = 2 * TINY.n_embd ** 2
check("MoE active = GPT active + router", abs(active_moe - active_gpt - router_cost) < 1,
      f"gpt {active_gpt:,.0f}, moe {active_moe:,.0f}")
check("maps add 2 x C^2 active", abs(active_maps - active_moe - map_cost) < 1, f"moe+maps {active_maps:,.0f}")

# ---- 7. embedding maps: identity changes nothing; a fit recovers a linear relation ------------------------
with torch.no_grad():
    d = max_diff(moe_maps(idx, route=route), mm.build(header, experts[0], experts).eval()(idx, route=route))
check("identity maps change nothing", d < 1e-4, f"max diff {d:.1e}")
C = TINY.n_embd
W, P = moe_maps.transformer.wte.weight.detach(), moe_maps.transformer.wpe.weight.detach()
R = [torch.randn(C, C) / C ** 0.5 for _ in range(E)]
counts = [torch.randint(0, 100, (gpt.VOCAB,)).double() for _ in range(E)]
residuals = mm.fit_embedding_maps(moe_maps, [W @ r for r in R], [P @ r for r in R], token_counts=counts)
err = max(max_diff(moe_maps.embed_maps[e], R[e]) for e in range(E))
check("map fit recovers an exact linear relation", max(residuals) < 1e-3 and err < 1e-2,
      f"residual {max(residuals):.1e}, max |M - R| {err:.1e}")

# Frequent tokens follow R1, rare ones R2: the count-weighted fit should find R1, the unweighted one not.
R1, R2 = torch.randn(C, C) / C ** 0.5, torch.randn(C, C) / C ** 0.5
target = W @ R2
target[:200] = W[:200] @ R1
heavy = torch.ones(gpt.VOCAB, dtype=torch.float64)
heavy[:200] = 1e6
mm.fit_embedding_maps(moe_maps, [target] * E, [P @ R1] * E, token_counts=[heavy] * E)
weighted = ((moe_maps.embed_maps[0] - R1).norm() / R1.norm()).item()
mm.fit_embedding_maps(moe_maps, [target] * E, [P @ R1] * E)
uniform = ((moe_maps.embed_maps[0] - R1).norm() / R1.norm()).item()
check("token counts steer the map fit to the frequent tokens", weighted < 0.05 and uniform > 0.5,
      f"|M - R1| / |R1|: weighted {weighted:.3f}, unweighted {uniform:.3f}")

# The closed-form residual (cross terms included) must equal the error measured on sampled (token, position)
# pairs, and match the best map fitted directly to those samples. Offsets make every mean non-zero.
m3 = mm.build(header, experts[1], experts, embed_map=True).eval()
with torch.no_grad():
    m3.transformer.wte.weight.add_(0.3)
    m3.transformer.wpe.weight.sub_(0.2)
Ws, Ps = m3.transformer.wte.weight.detach(), m3.transformer.wpe.weight.detach()
R0 = torch.randn(C, C) / C ** 0.5  # partly linear targets, so the residual is well inside (0, 1)
We = Ws @ R0 + 0.1 * torch.randn(gpt.VOCAB, C) + 0.05
Pe = Ps @ R0 + 0.1 * torch.randn(TINY.block_size, C) - 0.04
f = torch.rand(gpt.VOCAB).double() ** 4
r = mm.fit_embedding_maps(m3, [We] * E, [Pe] * E, token_counts=[f] * E)[0]
g = torch.Generator().manual_seed(1)
t = torch.multinomial(f.float(), 200_000, replacement=True, generator=g)
p = torch.randint(0, TINY.block_size, (200_000,), generator=g)
s, u = (Ws[t] + Ps[p]).double(), (We[t] + Pe[p]).double()
sampled = (((s @ m3.embed_maps[0].double() - u) ** 2).sum(1).mean() / (u ** 2).sum(1).mean()).sqrt().item()
M_direct = torch.linalg.lstsq(s, u).solution
direct = (((s @ M_direct - u) ** 2).sum(1).mean() / (u ** 2).sum(1).mean()).sqrt().item()
check("map residual formula matches sampled pairs", abs(r - sampled) < 0.01 and abs(r - direct) < 0.01,
      f"formula {r:.4f}, sampled {sampled:.4f}, direct fit on samples {direct:.4f}")

# ---- 8. save / load through model_def.py, as create_submission.py does --------------------------------------
with tempfile.TemporaryDirectory() as out:
    mm.save_solution(moe_maps, out)
    sys.path.insert(0, out)
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_def", os.path.join(out, "model_def.py"))
    model_def = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_def)
    loaded = model_def.load_model(out, "cpu")
    with torch.no_grad():
        d = max_diff(loaded(idx), moe_maps(idx))
    check("save_solution / load_model round trip", d == 0.0, f"max diff {d:.1e}")
    check("loaded count matches", official.count_parameters(loaded) == official.count_parameters(moe_maps))

# ---- 9. gradients and optimizer groups ------------------------------------------------------------------
moe.train()
logits = moe(idx, route=torch.tensor([0, 1]))
logits.float().logsumexp(-1).mean().backward()
grads = [moe.transformer.h[0].experts[e].c_fc.weight.grad for e in range(E)]
check("only routed experts get gradients",
      grads[0] is not None and grads[1] is not None and grads[2] is None and grads[3] is None)
check("router gets no gradient from the LM loss", moe.router.proj.weight.grad is None)
groups = gpt.optimizer_groups(moe, 0.1)
in_groups = sum(len(g["params"]) for g in groups)
hidden = {id(p) for g in groups if g["lr_mult"] != 1.0 for p in g["params"]}
check("optimizer_groups covers every parameter once", in_groups == len(list(moe.parameters())))
check("expert matrices are muP hidden", id(moe.transformer.h[0].experts[3].c_proj.weight) in hidden)

# ---- 10. full-size budget -------------------------------------------------------------------------------
for maps in (False, True):
    full = mm.MoEGPT(mm.MoEConfig(n_experts=4, embed_map=maps))
    total = official.count_parameters(full)
    check(f"full size {'with' if maps else 'without'} maps: total <= 300M", total <= official.MAX_PARAMETERS,
          f"{total / 1e6:.2f}M")
    if maps:
        active = official.active_parameters(full, torch.randint(0, gpt.VOCAB, (64,)), "cpu")
        check("full size active <= 150M", active <= 150e6, f"{active / 1e6:.2f}M")
    del full

print(f"\n{'all checks passed' if not FAILED else f'{len(FAILED)} FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
