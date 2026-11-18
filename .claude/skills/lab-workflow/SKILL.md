---
name: lab-workflow
description: How this team organizes, launches, records, syncs and visualizes experiments for the 10-605 expert-merging mini-project. Load before writing or editing a training script, launching or monitoring a run, touching an EC2 instance, moving files between laptop and EC2, recording or committing results, or building/reading the results dashboard. It covers process and bookkeeping only, not which experiments to run.
---

# Lab workflow

This skill covers *how* we do research here: where files go, how a run is launched and
recorded, how code and results move between laptops and EC2, and how progress is viewed. It
does **not** say which methods to try, which hyperparameters to use or what to conclude. Those
choices belong to the humans and the conversation. If something here conflicts with what the
user asks, ask the user.

Team: Hanmo (`hz`) and Ayush (`ay`). Each person's Claude works mostly from that person's laptop.

## 1. Know where you are

Run `./lab whereami` first in any session that will run, sync or launch something. It prints
`laptop` or `ec2:<host>`, the owner, and the default host.

- **Laptop:** edit code, commit, launch and monitor runs through `./lab`, view the dashboard,
  write notes, and commit results.
- **EC2:** only runs things. Do not edit tracked files there. The next `./lab push` overwrites
  them, and the run's recorded commit would be false. If a fix is needed, make it on the laptop
  and push again.
- Never write `ssh`/`scp`/`rsync` commands with raw IPs or hostnames. IPs change on every
  instance restart. Always go through `./lab` (it looks up the instance by ID).

## 2. Repository map

```
code/ solutions/ dev/ MANIFEST.sha256 README   course starter: never edit (merged from `upstream`)
models/ pool/                                   downloaded data, gitignored, EC2 only
lib/                                            shared helpers: launcher, metrics logging, eval
training_scripts/<method>/                      one folder per method family
    README.md                                   one-line description, owner, status
    *.py                                        scripts; each takes --out and writes a solution dir
runs/<run_id>/                                  solution output dir: weights + model_def.py (EC2 only, gitignored)
results/<run_id>/                               the committed record of a run (small files only)
tools/dashboard/                                Streamlit dashboard over results/
lab                                             the CLI for everything in this skill
lab_hosts.toml                                  committed: EC2 hosts (instance id, region, user)
lab.local.toml                                  per laptop, gitignored: owner, ssh key, aws profile
```

Adding a method family means creating `training_scripts/<method>/README.md` before its first
run. The launcher rejects a `--method` with no folder. The README starts with `# <method>`, then
`Description:`, `Owner:` and `Status:` lines (planned / active / done / dropped), which the
dashboard's Methods tab reads; keep `Status:` current. Method and variant names are lowercase
`[a-z0-9-]`.

## 3. Lifecycle of a run

1. **Write** the script in `training_scripts/<method>/`. It must accept `--out DIR` and write a
   valid solution directory there (`model_def.py` + weights + `train.log`), like
   `solutions/short_training.py`. To get curves on the dashboard, call `lib.metrics.log(step=…,
   tokens=…, train_loss=…, **anything)`. It is a no-op outside the launcher.
2. **Commit** it. `./lab run` refuses a dirty tree unless you pass `--allow-dirty`, and then it
   saves the diff into `results/<run_id>/code.diff`. Use that for quick iteration only, never
   for a run that goes into the report.
3. **Launch** it from the laptop:
   ```
   ./lab run [host] --method moe --variant ridge-router --hypothesis "one sentence" \
       [--parent <run_id>] -- python training_scripts/moe/train.py --steps 2000 --out '$LAB_OUT'
   ```
   This pushes the code, checks the GPU is free, assigns a run ID, starts the run in a tmux
   session named after it, and prints the run ID. On success it evaluates on dev automatically
   (`--no-eval` skips this). `--hypothesis` is required: what the run is meant to show, in the
   user's terms. Ask the user if it isn't clear.
4. **Monitor** with `./lab runs [host]` and `./lab logs <run_id> [-f]`. Do not keep polling in a
   loop. Check when there is a reason to.
5. **Pull** with `./lab pull [host]`. It copies `results/` back (never weights) and creates a
   `notes.md` stub for finished runs that lack one.
6. **Record** the outcome in `results/<run_id>/notes.md`: what happened, whether it supports the
   hypothesis, and anything odd. Failed and abandoned runs get notes too, because the report
   needs ablations and unsuccessful results. Write the facts; leave interpretation to the user
   unless asked.
7. **Commit** `results/<run_id>/` together with any code changes, then `git push`. This is how
   the teammate and their dashboard see the run.

## 4. Naming

- Run ID: `<YYYYMMDD-HHMM UTC>_<method>_<variant>_<owner>`, e.g.
  `20261118-1930_moe_ridge-router_hz`. It is generated by `./lab`, never by hand. A run ID is
  never reused, and a `results/<run_id>/` directory is never overwritten or deleted. Mark a bad
  run with `"status"` and notes instead.
- Variant = what makes this run different from others of the same method (`top1-avg-init`,
  `kd-a0.3`). Keep it short; full settings go in `run.json`.
- Seeds, learning rates and lengths may change within a variant name. The course does not
  count them as separate methods.

## 5. What a run records (`results/<run_id>/`)

| File | Written by | Content |
|---|---|---|
| `run.json` | launcher (EC2) | run_id, method, variant, owner, host, hypothesis, parent, command, git commit, dirty flag, start/end UTC, wall-clock and A10G-hours, status (`running`/`done`/`failed`/`killed`), heartbeat, exit code, and after eval: per-domain dev loss, mean, total and active params, causal, valid |
| `metrics.jsonl` | `lib.metrics.log` | one JSON line per log call: step, tokens, elapsed_s, plus whatever the script logs |
| `stdout.log` | launcher | everything the command printed |
| `train.log`, `submission.log` | copied from `runs/<run_id>/` | the files Gradescope wants |
| `window_losses.npz` | eval | dev cross-entropy per 2,048-token window per domain (for paired bootstrap) |
| `code.diff` | launcher | only if launched with `--allow-dirty` |
| `notes.md` | laptop, by a human or Claude | outcome and observations; never written on EC2 |

A run whose heartbeat is older than 10 minutes while its status is `running` shows as `lost` on
the dashboard. Investigate it with `./lab logs`; do not edit `run.json` by hand.

## 6. `./lab` reference

First-time setup on a laptop (once per person): `brew install awscli && aws configure`;
`pip install streamlit plotly pandas numpy`; `cp lab.local.toml.example lab.local.toml` and fill
in owner tag, ssh key and default host; put your instance ID in `lab_hosts.toml` and commit it;
then `./lab setup <host>`. If `./lab` fails because a config value is missing, tell the user which
one. Don't guess instance IDs or key paths.

```
./lab whereami                     laptop or ec2:<host>, owner, default host
./lab hosts                        each host: state, IP, GPU in use, remote commit vs local HEAD
./lab start <host>                 start the instance and wait until it is running
./lab stop <host> --yes            stop it (refuses while a run is active)
./lab setup <host>                 idempotent first-time setup: torch 2.10.0, data download, host marker
./lab push [host]                  sync tracked code + .git to the host (excludes data, runs/, results/)
./lab run [host] … -- <cmd>        push + launch under the recorder (see §3); --wait blocks until it
                                   ends, so `./lab run --wait … && ./lab run --wait …` is a queue
./lab runs [host] [--owner hz]     runs on the host with status, step, heartbeat age, dev score
./lab logs <run_id> [-f]           tail stdout.log
./lab kill <run_id>                stop a run; status becomes killed
./lab eval <run_id>                (re)evaluate a finished run on dev
./lab pull [host]                  copy results/ from the host (no weights)
./lab ssh [host]                   interactive shell in the host's checkout
./lab dashboard                    open the dashboard on localhost
```

`[host]` defaults to `default_host` in `lab.local.toml`. `LAB_DRY_RUN=1 ./lab …` prints the
aws/ssh/rsync commands instead of running them, which is useful when something looks wrong. You may read the teammate's host
(`hosts`, `runs`, `logs`, `pull`). Launch, kill, stop or push on it only if the user explicitly
asks. Each owner has their own checkout on a host (`~/mld/<owner>`), so a shared instance works
the same way. The GPU check is what keeps two runs from colliding.

**Starting and stopping instances costs money and interrupts work.** Never `stop` an instance with
a running run, and confirm `start`/`stop` with the user unless they just asked for it.

## 7. Dashboard

`./lab dashboard` runs Streamlit on localhost over the local `results/`. It shows only what has
been pulled or pulled from git, so run `./lab pull` / `git pull` first, and the sidebar shows
how fresh each host's data is. It includes: the leaderboard (with the course baselines as
reference lines), dev loss vs A10G-hours, training curves, per-domain losses, a paired bootstrap
between any two runs, per-run detail (hypothesis, notes, command, commit, log tail), and a CSV
export for the report table.

To add a view, add a tab in `tools/dashboard/app.py`. It reads the files in §5 and nothing else.
If a view needs new data, add the field in the launcher or in `lib.metrics` so every future run
records it. Charts follow the dataviz skill: methods keep a fixed color slot (alphabetical), so
filtering never recolors survivors. The bootstrap lives in `lib/bootstrap.py`, so report scripts
use the same numbers as the dashboard. Score the provided experts as baseline runs with variant
`expert-<domain>` (see `training_scripts/baseline/README.md`) and the Per domain tab marks each
domain's own-expert loss.

## 8. Course rules the bookkeeping protects

These restate the handout. They are not research advice.

- **Dev is for comparing only.** No script under `training_scripts/` reads `dev/` for training,
  tuning or early stopping. Only `./lab eval` / `code/create_submission.py` touch it. Tune on a
  held-out slice of `pool/`.
- **Only `pool/` data and provided models.** No downloads besides `code/download_data.py`.
- **Final recipe:** one script in `training_scripts/final/` that goes from the provided files to
  its `--out` directory in under 2 hours on a fresh g5.xlarge, with no network. It must run
  standalone (without `./lab`). Carry-over from earlier runs is only settings written into the
  script (≤ 1 MB), never weights or cached outputs. Fix all seeds.
- **Limits:** ≤ 300M total params, ≤ 150M active, causal. Eval records all three. Flag any run
  that breaks them in its notes.

## 9. Don'ts

- Don't create ad-hoc result files (CSV, screenshots, notebooks) outside `results/<run_id>/`. If a
  number matters, it belongs in the run record or the dashboard.
- Don't commit anything from `runs/`, `models/` or `pool/`, or any file over ~5 MB.
- Don't edit, move or delete another owner's `results/` or `training_scripts/` files without the
  user asking.
- Don't edit `code/`, `solutions/` or `dev/`. Copy what you need into `lib/` or your script.
