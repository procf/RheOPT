"""Trivial reference: error of persistence (every predicted frame equals the last input frame).

Validation cases as the models see them (every second stored frame; input window 2 frames for the contraction, 10 for
the wake), per-case relative L2 of velocity and stress over the predicted frames, averaged over cases; the wake uses
the 162 clean validation cases.

    python analysis/persistence_baseline.py
Output: persistence_baseline.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs  # noqa: E402
from rheopt.data import FlowDataset  # noqa: E402


def rel(p, t):
    return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=runs.RESULTS)
    a = ap.parse_args()
    rows = []
    clean = runs.wake_clean_files()
    for geo, fluid, tw, scaling in [("contraction", f, 2, "non_newtonian") for f in runs.FLUIDS] + [
            ("wake", "Oldroyd", 10, "all")]:
        ds = FlowDataset(os.path.join(runs.DATA, geo, fluid, "validation"), tw, velocity_scaling=scaling)
        ev, es = [], []
        for i, f in enumerate(ds.fname_lst):
            if geo == "wake" and os.path.basename(f) not in clean:
                continue
            x, y = ds[i][0].astype(np.float64), ds[i][1].astype(np.float64)
            p = np.repeat(x[-1:, :, :5], len(y), axis=0)
            ev.append(rel(p[..., :2], y[..., :2]))
            es.append(rel(p[..., 2:], y[..., 2:]))
        rows.append(dict(geometry=geo, fluid=fluid, n_cases=len(ev), frames_predicted=len(y),
                         vel_persistence=round(float(np.mean(ev)), 4), stress_persistence=round(float(np.mean(es)), 4)))
        print(rows[-1], flush=True)
    runs.write_csv(os.path.join(a.out, "persistence_baseline.csv"), rows)


if __name__ == "__main__":
    main()
