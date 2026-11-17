# training_scripts/

One folder per method family. A folder needs a `README.md` before its first run (`./lab run` checks),
starting with this header so the dashboard's Methods view can read it:

```
# <method>
Description: one line on what the method is
Owner: hz | ay | both
Status: planned | active | done | dropped
```

Every script takes `--out DIR` and writes a solution directory there (model_def.py, weights, train.log),
like `solutions/short_training.py`. Launch with `./lab run --method <folder> ... -- python training_scripts/<folder>/<script>.py --out '$LAB_OUT'`.
