"""Collect the metrics.json of every run under a runs folder into one table (setup, fluid, N, arm, seed, mean velocity and stress errors).

Run folders are found recursively; setup, fluid, N, arm and seed are read from the standard layout
<setup>/<fluid>/N<n>/<arm>_seed<seed>/ when it applies (custom run_dir values keep only the path). For the wake, the
reported means use the 162 clean validation cases.

    python analysis/summarize_runs.py [runs_folder] [--out results/runs_summary.csv]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs  # noqa: E402

LAYOUT = re.compile(r"(?P<setup>[^/]+)/(?P<fluid>[^/]+)/N(?P<N>\d+)/(?P<arm>finetune|scratch)_seed(?P<seed>\d+)$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_folder", nargs="?", default=runs.RUNS)
    ap.add_argument("--out", default=os.path.join(runs.RESULTS, "runs_summary.csv"))
    a = ap.parse_args()
    clean = runs.wake_clean_files()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.runs_folder, "**", "metrics.json"), recursive=True)):
        rel_dir = os.path.relpath(os.path.dirname(path), a.runs_folder)
        m = json.load(open(path))
        ps = m["per_sample"]
        sel = np.ones(len(ps["file"]), bool)
        if "wake" in rel_dir:
            sel = np.isin(ps["file"], list(clean))
        meta = LAYOUT.search(rel_dir)
        row = dict(run=rel_dir, **(meta.groupdict() if meta else dict(setup="", fluid="", N="", arm="", seed="")))
        row.update(n_val=len(sel), n_reported=int(sel.sum()),
                   vel_rel_l2=float(np.mean(np.array(ps["vel_rel_l2"])[sel])),
                   stress_rel_l2=float(np.mean(np.array(ps["stress_rel_l2"])[sel])),
                   vel_rel_l2_all_cases=m["velocity"]["mean"], stress_rel_l2_all_cases=m["stress"]["mean"])
        rows.append(row)
    if not rows:
        sys.exit(f"no metrics.json under {a.runs_folder}")
    runs.write_csv(a.out, rows)


if __name__ == "__main__":
    main()
