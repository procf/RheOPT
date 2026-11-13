"""Locating and reading run outputs (used by the analysis scripts).

Runs written by ``python -m rheopt`` live in ``<runs_root>/<setup>/<fluid>/N<n>/<arm>_seed<seed>/`` with
``metrics.json`` and ``predictions.npz``. Setting the environment variable ``RHEOPT_LEGACY`` to the folder of the
original runs (``<Fluid>/<variant>/N<n>/results/<mode>_N<n>_S<seed>.npz``) reads those instead.
"""
import csv
import glob
import os

import numpy as np

RUNS = os.environ.get("RHEOPT_RUNS", "runs")
DATA = os.environ.get("RHEOPT_DATA", "data")
RESULTS = os.environ.get("RHEOPT_RESULTS", "results")     # output folder of the analysis scripts
CLEAN_WAKE_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "wake_validation_clean.txt")
LEGACY = os.environ.get("RHEOPT_LEGACY")

SETUPS = ("contraction_revin_2s", "contraction_global_2s", "contraction_revin_10s", "contraction_global_10s",
          "wake_global", "wake_revin")
FLUIDS = ("Giesekus", "Oldroyd", "Bingham")
LABEL = {"Giesekus": "Giesekus", "Oldroyd": "Oldroyd-B", "Bingham": "Bingham"}
N_VALUES = {"contraction": (4, 8, 12, 16), "wake": (4, 8, 12, 16, 24, 32)}

_LEGACY_VARIANT = {"contraction_revin_2s": "RevIN-2s", "contraction_global_2s": "Standard-2s",
                   "contraction_revin_10s": "RevIN", "contraction_global_10s": "Standard",
                   "wake_revin": "RevIN", "wake_global": "Standard"}
_LEGACY_ARM = {"finetune": "fine_tune", "scratch": "train_only"}


def geometry(setup):
    return setup.split("_")[0]


def run_dir(setup, fluid, n, arm, seed):
    return os.path.join(RUNS, setup, fluid, f"N{n}", f"{arm}_seed{seed}")


def valid_dir(setup, fluid):
    return os.path.join(DATA, geometry(setup), fluid, "validation")


def load_predictions(setup, fluid, n, arm, seed):
    """Predictions of one run: dict with pos (cases, nodes, 2), gt and pred (cases, frames, nodes, 5; vx, vy, sxy,
    sxx, syy in cm/s and Pa), variables (cases, 3: Re, Wi or Bi, u_in) and files (validation file names)."""
    if LEGACY:
        folder = "Triangle_Oldroyd" if geometry(setup) == "wake" else fluid
        mode = _LEGACY_ARM[arm]
        d = dict(np.load(f"{LEGACY}/{folder}/{_LEGACY_VARIANT[setup]}/N{n}/results/{mode}_N{n}_S{seed}.npz"))
        d["variables"] = d.pop("Re")
        # the original runs read the validation folder in directory order, the order of this listing
        d["files"] = np.array([os.path.basename(f) for f in glob.glob(valid_dir(setup, fluid) + "/*.npz")])
        return d
    return dict(np.load(os.path.join(run_dir(setup, fluid, n, arm, seed), "predictions.npz")))


def wake_clean_files():
    """The 162 wake validation cases that are not also fine-tuning cases (the reported wake metrics use these)."""
    with open(CLEAN_WAKE_LIST) as fh:
        return {line.strip() for line in fh if line.strip()}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
