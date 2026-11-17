"""
The lab dashboard: every run in results/, on localhost.

    ./lab dashboard        # = streamlit run tools/dashboard/app.py -- --results results

It reads only the files a run records (lab-workflow skill, section 5), so it shows what has been pulled
(./lab pull) or pulled from git; the sidebar says how fresh each host's data is. To add a view, add a tab
below; if it needs new data, record it in lib/launch.py or lib/metrics.py so every run has it.
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lib.bootstrap import paired_bootstrap  # noqa: E402
from lib.common import DOMAINS, METHODS, REFERENCE, effective_status, read_json  # noqa: E402

# Reference palette (dataviz skill): categorical slots in fixed order, per theme; text never wears these.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
          "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
INK = {"light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e8e7e3"},
       "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#8a8984", "grid": "#333331"}}
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "pentagon", "star", "hexagon"]
STATUS_LABEL = {"running": "● running", "evaluating": "● evaluating", "done": "✓ done", "failed": "✕ failed",
                "killed": "■ killed", "lost": "? lost (no heartbeat)"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    return parser.parse_known_args()[0]


ARGS = parse_args()
st.set_page_config(page_title="Merge lab", page_icon="🧪", layout="wide")
THEME = getattr(getattr(st.context, "theme", None), "type", None) or "light"
PAL, INKS = SERIES[THEME], INK[THEME]


# ---- data ---------------------------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_runs(results):
    rows = []
    for path in glob.glob(os.path.join(results, "*", "run.json")):
        r = read_json(path, {})
        if not r.get("run_id"):
            continue
        ev, progress = r.get("eval") or {}, r.get("progress") or {}
        losses = ev.get("losses") or {}
        folder = os.path.dirname(path)
        rows.append({
            "run_id": r["run_id"], "method": r.get("method"), "variant": r.get("variant"), "owner": r.get("owner"),
            "status": effective_status(r), "host": r.get("host"), "started": r.get("started"),
            "a10g_hours": r.get("a10g_hours"), "dev": ev.get("mean"),
            **{d: losses.get(d) for d in DOMAINS},
            "params_m": ev["parameters"] / 1e6 if "parameters" in ev else None,
            "active_m": ev["active_parameters"] / 1e6 if "active_parameters" in ev else None,
            "valid": ev.get("valid"), "causal": ev.get("causal"), "eval_error": ev.get("error"),
            "commit": (r.get("git_commit") or "")[:8], "dirty": r.get("git_dirty"),
            "hypothesis": r.get("hypothesis"), "parents": ", ".join(r.get("parents") or []),
            "step": progress.get("step"), "heartbeat": r.get("heartbeat"),
            "has_windows": os.path.exists(os.path.join(folder, "window_losses.npz")),
            "has_metrics": os.path.exists(os.path.join(folder, "metrics.jsonl")),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("started", ascending=False, na_position="last").reset_index(drop=True)
    return df


@st.cache_data(ttl=5)
def load_metrics(results, run_id):
    path = os.path.join(results, run_id, "metrics.jsonl")
    lines = []
    with open(path) as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except ValueError:
                pass
    df = pd.DataFrame(lines)
    if "elapsed_s" in df:
        df["hours"] = df["elapsed_s"] / 3600
    return df


@st.cache_data
def load_windows(results, run_id, _mtime):
    with np.load(os.path.join(results, run_id, "window_losses.npz")) as z:
        return {d: z[d] for d in DOMAINS}


def windows(run_id):
    path = os.path.join(ARGS.results, run_id, "window_losses.npz")
    return load_windows(ARGS.results, run_id, os.path.getmtime(path))


def method_info():
    """The header of each training_scripts/<method>/README.md (Description, Owner, Status)."""
    info = {}
    for readme in sorted(glob.glob(os.path.join(METHODS, "*", "README.md"))):
        name, fields = os.path.basename(os.path.dirname(readme)), {}
        with open(readme) as f:
            for line in f.readlines()[:8]:
                key, _, value = line.partition(":")
                if key.strip() in ("Description", "Owner", "Status"):
                    fields[key.strip().lower()] = value.strip()
        info[name] = fields
    return info


def text_file(run_id, name, tail=None):
    path = os.path.join(ARGS.results, run_id, name)
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:] if tail else lines)


# ---- chart helpers ------------------------------------------------------------------------------------

def styled(fig, height=420, **layout):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INKS["secondary"], size=13), margin=dict(l=8, r=8, t=40, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(font_size=13), **layout)
    fig.update_xaxes(gridcolor=INKS["grid"], zeroline=False, linecolor=INKS["grid"])
    fig.update_yaxes(gridcolor=INKS["grid"], zeroline=False, linecolor=INKS["grid"])
    return fig


def reference_lines(fig):
    for name, value in REFERENCE.items():
        fig.add_hline(y=value, line=dict(color=INKS["muted"], width=1, dash="dot"),
                      annotation=dict(text=f"{name} {value:.3f}", font=dict(color=INKS["muted"], size=11)),
                      annotation_position="top left")


def stable_colors(key, selected):
    """Slot per selected run that survives other runs being added or removed (color follows the entity)."""
    slots = st.session_state.setdefault(f"slots_{key}", {})
    for run in [r for r in slots if r not in selected]:
        del slots[run]
    for run in selected:
        if run not in slots:
            slots[run] = min(set(range(len(PAL))) - set(slots.values()))
    return {run: PAL[slots[run]] for run in selected}


def fmt(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def age(stamp):
    if not stamp:
        return "-"
    s = time.time() - pd.Timestamp(stamp).timestamp()
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


# ---- page ---------------------------------------------------------------------------------------------

df_all = load_runs(ARGS.results)
methods = method_info()
all_methods = sorted(set(methods) | set(df_all["method"].dropna() if not df_all.empty else []))
METHOD_COLOR = {m: PAL[i] if i < len(PAL) else INKS["muted"] for i, m in enumerate(all_methods)}
METHOD_SYMBOL = {m: SYMBOLS[i % len(SYMBOLS)] for i, m in enumerate(all_methods)}

with st.sidebar:
    st.header("Merge lab")
    shown = os.path.relpath(ARGS.results, ROOT) if os.path.abspath(ARGS.results).startswith(ROOT) else ARGS.results
    st.caption(f"Results: `{shown}`")
    syncs = [read_json(p, {}) for p in sorted(glob.glob(os.path.join(ARGS.results, ".sync", "*.json")))]
    if syncs:
        for s in syncs:
            st.caption(f"{s['host']}: pulled {age(pd.Timestamp(s['pulled'], unit='s').isoformat())} ago")
    else:
        st.caption("No `./lab pull` yet on this laptop; showing what git has.")
    if st.button("Reload", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    if not df_all.empty:
        pick_methods = st.multiselect("Methods", all_methods, default=all_methods)
        pick_owners = st.multiselect("Owners", sorted(df_all["owner"].dropna().unique()),
                                     default=sorted(df_all["owner"].dropna().unique()))
        statuses = sorted(df_all["status"].unique())
        pick_status = st.multiselect("Status", statuses, default=statuses)
        dirty_ok = st.toggle("Include runs of uncommitted code", value=True)

st.title("Merge lab")
st.caption("One model from four experts: mean dev cross-entropy over math, code, medicine and history (nats, lower is better).")

if df_all.empty:
    st.info("No runs yet. Launch one with `./lab run`, then `./lab pull` (or `git pull` for your teammate's).")
    df = df_all
else:
    df = df_all[df_all["method"].isin(pick_methods) & df_all["owner"].isin(pick_owners)
                & df_all["status"].isin(pick_status) & (dirty_ok | ~df_all["dirty"].fillna(False))]

scored = df[df["dev"].notna()] if not df.empty else df
gen_runs = df_all[(df_all["method"] == "baseline") & (df_all["variant"] == "generalist") & df_all["dev"].notna()] \
    if not df_all.empty else df_all
GENERALIST = float(gen_runs["dev"].iloc[0]) if len(gen_runs) else REFERENCE["generalist"]
ORACLE = REFERENCE["per-domain experts (oracle)"]

# KPI row
k = st.columns(4)
if len(scored):
    best = scored.loc[scored["dev"].idxmin()]
    k[0].metric("Best dev loss", fmt(best["dev"]), f"{best['dev'] - GENERALIST:+.4f} vs generalist",
                delta_color="inverse", help=best["run_id"])
    k[1].metric("Gap closed", f"{(GENERALIST - best['dev']) / (GENERALIST - ORACLE) * 100:.1f}%",
                help="(generalist - best) / (generalist - per-domain experts)")
else:
    k[0].metric("Best dev loss", "-")
    k[1].metric("Gap closed", "-")
k[2].metric("Runs", f"{len(df)}", f"{int((df['status'] == 'running').sum()) if len(df) else 0} running",
            delta_color="off")
k[3].metric("A10G-hours spent", fmt(df["a10g_hours"].sum() if len(df) else 0, 1))

tabs = st.tabs(["Leaderboard", "Loss vs compute", "Training curves", "Per domain", "Compare", "Run detail",
                "Methods"])

# ---- Leaderboard ----
with tabs[0]:
    if df.empty:
        st.write("Nothing matches the filters.")
    else:
        board = df.copy()
        board["status"] = board["status"].map(lambda s: STATUS_LABEL.get(s, s))
        board["ppl"] = np.exp(board["dev"])
        board["Δ vs generalist"] = board["dev"] - GENERALIST
        board = board.sort_values(["dev", "started"], ascending=[True, False], na_position="last")
        cols = ["run_id", "method", "variant", "owner", "status", "dev", "Δ vs generalist", "ppl", *DOMAINS,
                "params_m", "active_m", "a10g_hours", "valid", "commit", "dirty", "step"]
        num = lambda label, f="%.4f": st.column_config.NumberColumn(label, format=f)  # noqa: E731
        st.dataframe(board[cols], hide_index=True, width="stretch", height=min(640, 38 + 35 * len(board)),
                     column_config={"dev": num("dev CE"), "Δ vs generalist": num("Δ vs gen.", "%+.4f"),
                                    "ppl": num("PPL", "%.2f"), **{d: num(d) for d in DOMAINS},
                                    "params_m": num("params (M)", "%.1f"), "active_m": num("active (M)", "%.1f"),
                                    "a10g_hours": num("A10G-h", "%.2f")})
        st.caption(f"Δ vs generalist uses {'our measured generalist run' if len(gen_runs) else 'the handout value'} "
                   f"({GENERALIST:.4f}). Handout references: "
                   + ", ".join(f"{k} {v:.3f}" for k, v in REFERENCE.items()))
        report = board[board["dev"].notna()][["run_id", "method", "variant", "dev", "ppl", "Δ vs generalist",
                                              "params_m", "active_m", "a10g_hours"]]
        st.download_button("Download report table (CSV)", report.to_csv(index=False), "results_table.csv")

# ---- Loss vs compute ----
with tabs[1]:
    if scored.empty:
        st.write("No scored runs yet.")
    else:
        log_x = st.toggle("Log-scale A10G-hours", value=False)
        fig = go.Figure()
        for m in [m for m in all_methods if m in set(scored["method"])]:
            sub = scored[scored["method"] == m]
            fig.add_trace(go.Scatter(
                x=sub["a10g_hours"].fillna(0), y=sub["dev"], mode="markers", name=m,
                marker=dict(size=11, color=METHOD_COLOR[m], symbol=METHOD_SYMBOL[m],
                            line=dict(width=2, color="rgba(0,0,0,0)")),
                customdata=sub[["run_id", "variant"]],
                hovertemplate="<b>%{customdata[0]}</b><br>variant %{customdata[1]}<br>dev %{y:.4f}<br>"
                              "%{x:.2f} A10G-h<extra></extra>"))
        reference_lines(fig)
        fig.update_xaxes(title="training A10G-hours", type="log" if log_x else "linear")
        fig.update_yaxes(title="mean dev cross-entropy")
        st.plotly_chart(styled(fig, 520), width="stretch")
        st.caption("Training time only (evaluation excluded). Dotted lines: handout reference scores.")

# ---- Training curves ----
with tabs[2]:
    with_metrics = df[df["has_metrics"]] if len(df) else df
    if with_metrics.empty:
        st.write("No run has logged metrics. Call `lib.metrics.log(step=..., train_loss=...)` in the training script.")
    else:
        running = list(with_metrics[with_metrics["status"] == "running"]["run_id"])
        default = (running + [r for r in with_metrics["run_id"] if r not in running])[:3]
        pick = st.multiselect("Runs (up to 8)", list(with_metrics["run_id"]), default=default, max_selections=8,
                              key="curve_runs")
        frames = {r: load_metrics(ARGS.results, r) for r in pick}
        numeric = sorted({c for f in frames.values() for c in f.select_dtypes("number").columns})
        c1, c2 = st.columns(2)
        x_opts = [c for c in ("step", "tokens", "hours") if c in numeric] or numeric
        x = c1.selectbox("x", x_opts) if x_opts else None
        y_opts = [c for c in numeric if c not in ("step", "tokens", "hours", "elapsed_s")]
        y = c2.selectbox("y", y_opts, index=y_opts.index("train_loss") if "train_loss" in y_opts else 0) \
            if y_opts else None
        if x and y:
            colors = stable_colors("curves", pick)
            fig = go.Figure()
            for r, f in frames.items():
                if x in f and y in f:
                    fig.add_trace(go.Scatter(x=f[x], y=f[y], mode="lines", name=r, line=dict(width=2, color=colors[r]),
                                             hovertemplate=f"<b>{r}</b><br>{x} %{{x}}<br>{y} %{{y:.4f}}<extra></extra>"))
            fig.update_xaxes(title=x)
            fig.update_yaxes(title=y)
            st.plotly_chart(styled(fig, 480, hovermode="x unified"), width="stretch")

# ---- Per domain ----
with tabs[3]:
    if scored.empty:
        st.write("No scored runs yet.")
    else:
        pick = st.multiselect("Runs (up to 8)", list(scored.sort_values("dev")["run_id"]),
                              default=list(scored.sort_values("dev")["run_id"])[:3], max_selections=8,
                              key="domain_runs")
        colors = stable_colors("domains", pick)
        fig = go.Figure()
        experts = {d: df_all[(df_all["method"] == "baseline") & (df_all["variant"] == f"expert-{d}")][d].dropna()
                   for d in DOMAINS} if not df_all.empty else {}
        own = [(d, float(v.iloc[0])) for d, v in experts.items() if len(v)]
        if own:
            fig.add_trace(go.Scatter(x=[v for _, v in own], y=[d for d, _ in own], mode="markers", name="own expert",
                                     marker=dict(symbol="line-ns", size=22, line=dict(width=2, color=INKS["muted"])),
                                     hovertemplate="own expert on %{y}: %{x:.4f}<extra></extra>"))
        for r in pick:
            row = scored[scored["run_id"] == r].iloc[0]
            fig.add_trace(go.Scatter(x=[row[d] for d in DOMAINS], y=DOMAINS, mode="markers", name=r,
                                     marker=dict(size=12, color=colors[r]),
                                     hovertemplate=f"<b>{r}</b><br>%{{y}} %{{x:.4f}}<extra></extra>"))
        fig.update_xaxes(title="dev cross-entropy")
        fig.update_yaxes(categoryorder="array", categoryarray=DOMAINS[::-1])
        st.plotly_chart(styled(fig, 380), width="stretch")
        if not own:
            st.caption("Score the experts as baseline runs (variant `expert-<domain>`) to mark each domain's own-expert loss.")

# ---- Compare (paired bootstrap) ----
with tabs[4]:
    paired = df_all[df_all["has_windows"]] if not df_all.empty else df_all
    if len(paired) < 2:
        st.write("Needs two runs with `window_losses.npz` (written by `./lab eval`).")
    else:
        ids = list(paired.sort_values("dev")["run_id"])
        ref_default = gen_runs["run_id"].iloc[0] if len(gen_runs) and gen_runs["run_id"].iloc[0] in ids else ids[-1]
        c1, c2 = st.columns(2)
        a = c1.selectbox("Run A", ids, index=0)
        b = c2.selectbox("Reference B", ids, index=ids.index(ref_default))
        res = paired_bootstrap(windows(a), windows(b))
        m = res["mean"]
        st.metric("A − B, mean over domains", f"{m['delta']:+.4f}")
        st.caption(f"95% CI [{m['lo']:+.4f}, {m['hi']:+.4f}]")
        rows = ["mean", *DOMAINS]
        fig = go.Figure(go.Scatter(
            x=[res[r]["delta"] for r in rows], y=rows, mode="markers", marker=dict(size=11, color=PAL[0]),
            error_x=dict(type="data", symmetric=False, array=[res[r]["hi"] - res[r]["delta"] for r in rows],
                         arrayminus=[res[r]["delta"] - res[r]["lo"] for r in rows], color=PAL[0], thickness=2, width=6),
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>", showlegend=False))
        fig.add_vline(x=0, line=dict(color=INKS["muted"], width=1))
        fig.update_xaxes(title="loss difference A − B (negative: A is better)")
        fig.update_yaxes(categoryorder="array", categoryarray=rows[::-1])
        st.plotly_chart(styled(fig, 320), width="stretch")
        st.caption("Paired bootstrap over dev windows, stratified by domain, 10,000 resamples (lib/bootstrap.py).")
        if st.button(f"Every scored run vs {b}"):
            table = []
            for r in ids:
                if r != b:
                    x = paired_bootstrap(windows(r), windows(b))["mean"]
                    table.append({"run_id": r, "Δ": x["delta"], "lo": x["lo"], "hi": x["hi"]})
            t = pd.DataFrame(table).sort_values("Δ")
            st.dataframe(t, hide_index=True, width="stretch")
            st.download_button("Download (CSV)", t.to_csv(index=False), f"bootstrap_vs_{b}.csv")

# ---- Run detail ----
with tabs[5]:
    if df_all.empty:
        st.write("No runs yet.")
    else:
        run = st.selectbox("Run", list(df_all["run_id"]))
        rec = read_json(os.path.join(ARGS.results, run, "run.json"), {})
        row = df_all[df_all["run_id"] == run].iloc[0]
        st.subheader(run)
        c = st.columns(4)
        c[0].metric("Status", STATUS_LABEL.get(row["status"], row["status"]))
        c[1].metric("Dev loss", fmt(row["dev"]))
        c[2].metric("A10G-hours", fmt(row["a10g_hours"], 2))
        c[3].metric("Heartbeat", age(row["heartbeat"]) + " ago" if row["heartbeat"] else "-")
        st.markdown(f"**Hypothesis.** {rec.get('hypothesis', '')}")
        st.markdown(f"**Commit** `{rec.get('git_commit')}`{' (uncommitted changes, see code.diff)' if rec.get('git_dirty') else ''}"
                    f" · **host** {rec.get('host')} · **owner** {rec.get('owner')}"
                    f"{' · **parents** ' + ', '.join(rec['parents']) if rec.get('parents') else ''}")
        st.code(" ".join(rec.get("command") or []), language="bash")
        if row["eval_error"]:
            st.warning(f"Evaluation: {row['eval_error']}")
        notes = text_file(run, "notes.md")
        st.markdown("**Notes**")
        st.markdown(notes if notes else "_No notes.md yet: `./lab pull` creates a stub for finished runs._")
        with st.expander("stdout.log (last 80 lines)"):
            st.code(text_file(run, "stdout.log", tail=80) or "(empty)")
        for name in ("submission.log", "code.diff"):
            content = text_file(run, name)
            if content:
                with st.expander(name):
                    st.code(content, language="diff" if name.endswith(".diff") else None)
        with st.expander("run.json"):
            st.json(rec)

# ---- Methods ----
with tabs[6]:
    rows = []
    for m in all_methods:
        sub = df_all[df_all["method"] == m] if not df_all.empty else df_all
        sc = sub[sub["dev"].notna()] if len(sub) else sub
        best_row = sc.loc[sc["dev"].idxmin()] if len(sc) else None
        rows.append({"method": m, "description": methods.get(m, {}).get("description", "(no README)"),
                     "owner": methods.get(m, {}).get("owner", ""), "status": methods.get(m, {}).get("status", ""),
                     "runs": len(sub), "scored": len(sc),
                     "best dev": None if best_row is None else best_row["dev"],
                     "best run": "" if best_row is None else best_row["run_id"],
                     "A10G-h": float(sub["a10g_hours"].fillna(0).sum()) if len(sub) else 0.0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"best dev": st.column_config.NumberColumn(format="%.4f"),
                                "A10G-h": st.column_config.NumberColumn(format="%.2f")})
    st.caption("From each training_scripts/<method>/README.md header (Description, Owner, Status).")
