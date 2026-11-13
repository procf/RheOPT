"""Estimate of the solvent viscosity ratio of the viscoelastic contraction data: solvent / total shear stress in the
downstream channel (4 < x < 8, y > 0.5), at the last stored frame, where the total shear stress exceeds 20 % of its
maximum; median over nodes, then over the first 12 validation cases. Needs the polymer stress key ``tau_p_xy``.

    python analysis/viscosity_ratio.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt import runs  # noqa: E402

for fluid in ("Oldroyd", "Giesekus"):
    betas = []
    for f in sorted(glob.glob(os.path.join(runs.DATA, "contraction", fluid, "validation", "*.npz")))[:12]:
        d = np.load(f)
        pos = d["pos"][0] if d["pos"].ndim == 3 else d["pos"]
        m = (pos[:, 0] > 4) & (pos[:, 0] < 8) & (pos[:, 1] > 0.5)
        sxy, txy = d["Sigma_xy"][-1][m], d["tau_p_xy"][-1][m, 0]
        ok = np.abs(sxy) > 0.2 * np.abs(sxy).max()
        betas.append(np.median((sxy[ok] - txy[ok]) / sxy[ok]))
    print(f"{runs.LABEL[fluid]}: solvent / total shear stress = {np.median(betas):.3f} "
          f"(range {min(betas):.3f}-{max(betas):.3f}, {len(betas)} cases)")
