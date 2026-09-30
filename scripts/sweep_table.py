"""Summarise the hyper-parameter sweep (validation -log PL per sequence) -> outputs/sweep.json."""
import glob
import json

import numpy as np

from common import OUT

rows = []
for f in sorted(glob.glob(str(OUT / "train_hist_sweep_*.json"))):
    h = json.load(open(f))
    c = h["cfg"]
    if c["epochs"] < 3000:          # skip runs from the aborted 500-epoch sweep
        continue
    rows.append({"H": c["H"], "d": c["d"], "lambd": c["lambd"], "val": h["best_val_nlpl"],
                 "best_epoch": int(np.argmin(h["val_nlpl"]) + 1), "n_params": h["n_params"],
                 "tag": c["tag"]})
rows.sort(key=lambda r: r["val"])
for r in rows:
    print(f"H={r['H']:3d} d={r['d']:3d} lambda={r['lambd']:.0e}  val {r['val']:.3f}  "
          f"best epoch {r['best_epoch']:5d}  params {r['n_params']}")
json.dump(rows, open(OUT / "sweep.json", "w"), indent=1)
