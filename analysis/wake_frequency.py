"""Wake diagnostics per run: recovery of the shedding frequency and RMS of v_y for the representative case.

For every run of the wake setups (seeds as given), the dominant frequency of v_x in the wake is computed for the
ground truth and the prediction of each validation case. A case sheds when its ground-truth peak is above the lowest
non-zero bin (k >= 2). Recovery fractions are over the shedding cases among the 162 clean validation cases.

    python analysis/wake_frequency.py [--seeds 1 2 3 4] [--out results]
Outputs: wake_frequency.csv (per run), wake_peak_bins.csv (per case), wake_rms.csv (RMS profiles, Re = 94.1 case)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs, wake  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", default=["wake_global", "wake_revin"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", default=runs.RESULTS)
    a = ap.parse_args()
    clean = runs.wake_clean_files()
    per_run, per_case, rms_rows = [], [], []
    for setup in a.setups:
        for n in runs.N_VALUES["wake"]:
            for seed in a.seeds:
                for arm in ("finetune", "scratch"):
                    d = runs.load_predictions(setup, "Oldroyd", n, arm, seed)
                    pos, re = d["pos"][0], d["variables"][:, 0]
                    m = wake.wake_mask(pos)
                    k_gt = np.array([wake.peak_bin(s[:, m, 0]) for s in d["gt"]])
                    k_pr = np.array([wake.peak_bin(s[:, m, 0]) for s in d["pred"]])
                    is_clean = np.isin(d["files"], list(clean))
                    shed = (k_gt >= 2) & is_clean
                    per_run.append(dict(setup=setup, N=n, seed=seed, arm=arm, n_cases=int(is_clean.sum()),
                                        n_shedding=int(shed.sum()),
                                        frac_exact_bin=round(float((k_pr[shed] == k_gt[shed]).mean()), 4),
                                        frac_within_1_bin=round(float((abs(k_pr[shed] - k_gt[shed]) <= 1).mean()), 4),
                                        frac_no_oscillation=round(float((k_pr[shed] == 1).mean()), 4),
                                        bin_width_Hz=round(1 / (d["gt"].shape[1] * wake.DT), 4)))
                    per_case += [dict(setup=setup, N=n, seed=seed, arm=arm, file=f, Re=round(float(r), 4),
                                      clean=int(c), k_gt=int(g), k_pred=int(p))
                                 for f, r, c, g, p in zip(d["files"], re, is_clean, k_gt, k_pr)]
                    i = int(np.argmin(np.abs(re - wake.REPRESENTATIVE_RE)))
                    for label, field in (("prediction", d["pred"][i]), ("ground_truth", d["gt"][i])):
                        x, r = wake.rms_profile(field[:, :, 1], pos[:, 0])
                        rms_rows += [dict(setup=setup, N=n, seed=seed, arm=arm, field=label, x=round(float(xx), 3),
                                          rms=round(float(v), 5)) for xx, v in zip(x, r)]
                    print(per_run[-1], flush=True)
    runs.write_csv(os.path.join(a.out, "wake_frequency.csv"), per_run)
    runs.write_csv(os.path.join(a.out, "wake_peak_bins.csv"), per_case)
    runs.write_csv(os.path.join(a.out, "wake_rms.csv"), rms_rows)


if __name__ == "__main__":
    main()
