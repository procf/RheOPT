"""Size of each field per simulation: mean and s.d. of every channel over the whole trajectory, the input window and
the predicted window (in the units the models see: cm/s and Pa). Used for the normalization discussion (how much the
scale of the fields varies between simulations and between the input and predicted windows).

    python analysis/field_statistics.py
Output: field_statistics.csv
"""
import glob
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs  # noqa: E402
from rheopt.data import CHANNELS, FlowDataset  # noqa: E402

DATASETS = [("contraction", "Newtonian", 2, "non_newtonian")] + \
           [("contraction", f, 2, "non_newtonian") for f in runs.FLUIDS] + \
           [("wake", "Newtonian", 10, "all"), ("wake", "Oldroyd", 10, "all")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=runs.RESULTS)
    a = ap.parse_args()
    rows = []
    for geo, fluid, tw, scaling in DATASETS:
        root = os.path.join(runs.DATA, geo, fluid)
        folders = [root] if glob.glob(root + "/*.npz") else sorted(d for d in glob.glob(root + "/*") if os.path.isdir(d))
        seen = set()
        for folder in folders:
            ds = FlowDataset(folder, tw, velocity_scaling=scaling)
            for i, f in enumerate(ds.fname_lst):
                name = os.path.basename(f)
                if name in seen:           # the same simulation can sit in several fine-tuning folders
                    continue
                seen.add(name)
                x, y, _, _, var = ds[i]
                full = np.concatenate([x[..., :5], y], axis=0).astype(np.float64)
                for c, ch in enumerate(CHANNELS):
                    rows.append(dict(geometry=geo, fluid=fluid, file=name, Re=round(float(var[0]), 4),
                                     second=round(float(var[1]), 4), channel=ch,
                                     mean_full=float(full[..., c].mean()), std_full=float(full[..., c].std()),
                                     mean_in=float(full[:tw, :, c].mean()), std_in=float(full[:tw, :, c].std()),
                                     mean_pred=float(full[tw:, :, c].mean()), std_pred=float(full[tw:, :, c].std()),
                                     absmax_full=float(np.abs(full[..., c]).max())))
        print(f"{geo} {fluid}: {len(seen)} simulations", flush=True)
    runs.write_csv(os.path.join(a.out, "field_statistics.csv"), rows)


if __name__ == "__main__":
    main()
