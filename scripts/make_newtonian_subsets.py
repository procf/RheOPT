"""Nested, Re-stratified subsets of the Newtonian contraction simulations for the pretraining-size experiment (E2).

The 200 files are sorted by Re (file name Re<value>.npz); N_pre = 100, 50, 25 take every 2nd, 4th, 8th file starting
with the lowest Re. Writes symlink folders data/contraction/Newtonian_subsets/Npre<k>/.
    python scripts/make_newtonian_subsets.py [data_root]
"""
import glob
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RHEOPT_DATA", "data")
src = os.path.join(root, "contraction", "Newtonian")
files = sorted(glob.glob(f"{src}/*.npz"), key=lambda f: float(re.search(r"Re([0-9.]+)\.npz$", f).group(1)))
assert len(files) == 200, len(files)
for npre, step in ((100, 2), (50, 4), (25, 8)):
    out = os.path.join(root, "contraction", "Newtonian_subsets", f"Npre{npre}")
    os.makedirs(out, exist_ok=False)
    for f in files[::step]:
        os.symlink(os.path.abspath(f), os.path.join(out, os.path.basename(f)))
    print(out, len(files[::step]), "files")
