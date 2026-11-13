"""Stress error split into its components, per contraction run.

The stress channels are total stresses (sigma = -p I + tau); the reported stress error (relative L2 over sxy, sxx, syy
together) is dominated by the pressure in the normal stresses. Per case, then averaged over cases:
    stress       relative L2 over the three stress channels (the reported metric)
    sxy          shear stress alone
    n1           first normal-stress difference N1 = sxx - syy (pressure-free)
    mean_normal  (sxx + syy) / 2 (pressure-dominated)
and pooled over all cases (sqrt of summed squared error over summed squared truth): velocity, stress, sxy, N1, plus
the size of N1 relative to the normal stresses in the ground truth (median and maximum over cases).

    python analysis/stress_components.py [--setups contraction_revin_2s contraction_global_2s] [--seeds 1 2 3 4]
Output: stress_components.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs  # noqa: E402


def rel(p, g):
    return np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-8)


def pooled(p, g):
    return np.sqrt(np.sum((p - g) ** 2) / np.sum(g ** 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=["contraction_revin_2s", "contraction_global_2s"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", default=runs.RESULTS)
    a = ap.parse_args()
    rows = []
    for setup in a.setups:
        for fluid in runs.FLUIDS:
            for n in runs.N_VALUES["contraction"]:
                for seed in a.seeds:
                    for arm in ("finetune", "scratch"):
                        d = runs.load_predictions(setup, fluid, n, arm, seed)
                        g, p = d["gt"].astype(np.float64), d["pred"].astype(np.float64)
                        n1g, n1p = g[..., 3] - g[..., 4], p[..., 3] - p[..., 4]
                        per_case = {
                            "stress": [rel(p[i, ..., 2:], g[i, ..., 2:]) for i in range(len(g))],
                            "sxy": [rel(p[i, ..., 2], g[i, ..., 2]) for i in range(len(g))],
                            "n1": [rel(n1p[i], n1g[i]) for i in range(len(g))],
                            "mean_normal": [rel(p[i, ..., 3:5].mean(-1), g[i, ..., 3:5].mean(-1)) for i in range(len(g))],
                        }
                        size = [np.linalg.norm(n1g[i]) / np.linalg.norm(g[i, ..., 3:5]) for i in range(len(g))]
                        row = dict(setup=setup, fluid=fluid, N=n, arm=arm, seed=seed, n_cases=len(g))
                        row.update({k: round(float(np.mean(v)), 4) for k, v in per_case.items()})
                        row.update(vel_pooled=round(pooled(p[..., :2], g[..., :2]), 4),
                                   stress_pooled=round(pooled(p[..., 2:], g[..., 2:]), 4),
                                   sxy_pooled=round(pooled(p[..., 2], g[..., 2]), 4),
                                   n1_pooled=round(pooled(n1p, n1g), 4),
                                   n1_size_median=round(float(np.median(size)), 4),
                                   n1_size_max=round(float(np.max(size)), 4))
                        rows.append(row)
                        print(row, flush=True)
    runs.write_csv(os.path.join(a.out, "stress_components.csv"), rows)


if __name__ == "__main__":
    main()
