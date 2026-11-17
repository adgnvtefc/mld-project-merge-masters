"""
Baseline: one provided checkpoint, unchanged, as a solution directory (no training).

    python3 training_scripts/baseline/provided_model.py --model generalist --out runs/x
"""

import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["seed", "generalist", "expert_math", "expert_code", "expert_medicine", "expert_history"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    source = os.path.realpath(os.path.join(ROOT, "models", f"{args.model}.bin"))
    target = os.path.join(args.out, "model.bin")
    if os.path.lexists(target):
        os.remove(target)
    os.symlink(source, target)  # 250 MB each: link instead of copying
    shutil.copy(os.path.join(ROOT, "solutions", "gpt_model_def.py"), os.path.join(args.out, "model_def.py"))
    with open(os.path.join(args.out, "train.log"), "w") as f:
        f.write(f"no training: models/{args.model}.bin as provided\n")
    print(f"wrote {args.out}: model.bin -> {source}, model_def.py")


if __name__ == "__main__":
    main()
