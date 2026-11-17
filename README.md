# For humans

The course handout is `README`. Claude's rules live in `CLAUDE.md` and the lab-workflow skill.

## First time

- **Laptop setup**: Do once: `brew install awscli`, `aws configure`, `pip install streamlit plotly pandas`, then fill in `lab.local.toml`.
- **Your instance**: Add your EC2 instance ID to `lab_hosts.toml` so `./lab` can find your machine.
- **`./lab setup`**: Prepares a fresh EC2 instance (PyTorch, data download, folders); run `./lab setup hanmo` once per instance.

## Everyday commands

- **`./lab whereami`**: Tells you whether you're on your laptop or on EC2: `./lab whereami`.
- **`./lab hosts`**: Shows each EC2 machine's state, GPU use and code version: `./lab hosts`.
- **`./lab start` / `stop`**: Turns your instance on or off to save money: `./lab start hanmo`, `./lab stop hanmo --yes`.
- **`./lab push`**: Copies your code to EC2, which `./lab run` also does automatically: `./lab push`.
- **`./lab run`**: Launches and records an experiment: `./lab run --method M --variant V --hypothesis "..." -- python <script> --out '$LAB_OUT'`.
- **`./lab runs`**: Lists runs on EC2 with status, progress and dev score: `./lab runs`.
- **`./lab logs`**: Shows a run's live output: `./lab logs <run_id> -f`.
- **`./lab kill`**: Stops a running experiment: `./lab kill <run_id>`.
- **`./lab eval`**: Re-scores a finished run on the dev set: `./lab eval <run_id>`.
- **`./lab pull`**: Copies results (not weights) from EC2 to your laptop; then commit `results/` and push: `./lab pull`.
- **`./lab ssh`**: Opens a terminal in your checkout on EC2: `./lab ssh`.
- **`./lab dashboard`**: Opens a page with the leaderboard, plots and bootstrap comparisons: `./lab dashboard`, then visit `localhost:8605`.

## Folders

- **`training_scripts/`**: Your training code, one folder per method; add a `README.md` before a method's first run.
- **`results/`**: The committed record of every run; write your conclusions in each run's `notes.md`.
- **`runs/`**: Model weights, kept only on EC2 and never committed.
- **`lib/`**: Shared helpers; call `lib.metrics.log(step=..., train_loss=...)` in training scripts to get dashboard curves.
- **`code/`, `solutions/`, `dev/`**: The course's starter code and dev set; never edit them.
- **`tools/`**: The code behind `./lab` and the dashboard; you rarely need to open it.
- **`.claude/skills/lab-workflow/`**: The workflow rules Claude follows automatically; edit them to change how Claude organizes experiments.
