"""Flow datasets on unstructured meshes.

Each simulation is one ``.npz`` file with the keys

    pos        (n, 2)          node coordinates (cm)
    vel        (T, n, 2)       velocity, m/s for the non-Newtonian files, cm/s for the Newtonian contraction files
    Sigma_xy, Sigma_xx, Sigma_yy  (T, n)  total stress components
    Re         scalar          Reynolds number
    Wi or Bi   scalar          Weissenberg (Oldroyd-B, Giesekus) or Bingham number; absent for Newtonian files
    u_in / u_0 / u0  scalar    inlet velocity

A sample is (x, y, node_type, pos, variables):

    x          (tw, n, 7)      the first ``time_window`` frames of [vx, vy, sxy, sxx, syy] followed by (x, y)
    y          (T' - tw, n, 5) the remaining frames of [vx, vy, sxy, sxx, syy]
    node_type  (n, 1)          1 on the walls selected by ``wall_mask``, 0 elsewhere
    pos        (n, 2)          node coordinates
    variables  (3,)            [Re, Wi or Bi (0 for Newtonian), u_in]

Frames are taken every ``interval`` (= 2) stored frames. This reproduces the loader of the original experiments
(``AirfoilData`` in ``common/datasets.py``); the few ways in which its copies differed between experiments are the
``wall_mask`` and ``velocity_scaling`` options (see docs/PROTOCOL_NOTES.md).
"""
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CHANNELS = ("vx", "vy", "sxy", "sxx", "syy")

WALL_MASKS = ("contraction", "contraction_no_step", "triangle")
VELOCITY_SCALINGS = ("non_newtonian", "all")


def wall_mask(pos, kind):
    """Boolean wall indicator per node.

    contraction          y = 4 for x < 0 (upstream wall), y = 1 for x >= 0 (downstream wall) and the step face
                         (x > -0.001, y >= 1). Used for fine-tuning and training from scratch.
    contraction_no_step  the same without the step face. Used by the contraction and wake pretraining runs.
    triangle             the three edges of the triangular obstacle with vertices (-3, 0.1), (-3, -0.1), (-2.8, 0.1).
                         Used by the wake runs with Global normalization.
    """
    x, y = pos[:, 0], pos[:, 1]
    if kind in ("contraction", "contraction_no_step"):
        mask = ((y == 4) & (x < 0)) | ((y == 1) & (x >= 0))
        if kind == "contraction":
            mask = mask | ((x > -0.001) & (y >= 1))
        return mask
    if kind == "triangle":
        edge1 = np.isclose(x, -3.0, atol=5e-2) & (y >= -0.1 - 5e-2) & (y <= 0.1 + 5e-2)
        edge2 = np.isclose(y - x - 2.9, 0.0, atol=8e-2) & (x >= -3.0 - 3e-2) & (x <= -2.8 + 3e-2)
        edge3 = np.isclose(y, 0.1, atol=5e-2) & (x >= -3.0 - 5e-2) & (x <= -2.8 + 5e-2)
        return edge1 | edge2 | edge3
    raise ValueError(f"wall_mask must be one of {WALL_MASKS}, got {kind!r}")


class FlowDataset(Dataset):
    """All simulations (``*.npz``) in one folder, held in memory.

    Args:
        data_path: folder with the ``.npz`` files. Every file in it is used.
        time_window: number of input frames.
        wall_mask: see :func:`wall_mask`.
        velocity_scaling: ``non_newtonian`` multiplies the velocity by 100 (m/s to cm/s) only for files with ``Wi`` or
            ``Bi``; ``all`` multiplies every file's velocity and u_in by 100 (wake data).
        use_normalized: return fields normalized with this folder's global mean and standard deviation per channel
            (used for training from scratch and pretraining under Global normalization).
        interval: temporal subsampling of the stored frames.

    Files are read in ``glob`` order, as in the original runs; the order sets which simulations share a batch.
    """

    def __init__(self, data_path, time_window, wall_mask="contraction", velocity_scaling="non_newtonian",
                 use_normalized=False, interval=2):
        if velocity_scaling not in VELOCITY_SCALINGS:
            raise ValueError(f"velocity_scaling must be one of {VELOCITY_SCALINGS}")
        self.data_path = data_path
        self.tw = time_window
        self.wall_mask = wall_mask
        self.velocity_scaling = velocity_scaling
        self.use_normalized = use_normalized
        self.interval = interval
        self.fname_lst = glob.glob(data_path + "/*.npz")
        if not self.fname_lst:
            raise FileNotFoundError(f"no .npz files in {data_path}")
        self.cache = {}
        self._prepare()

    def __len__(self):
        return len(self.fname_lst)

    def _load(self, fname):
        data = np.load(fname)
        pos = data["pos"][:, :].astype(np.float32)
        if pos.ndim == 3:
            pos = pos[0, ...]
        node_type = torch.tensor(wall_mask(pos, self.wall_mask), dtype=torch.float32)[:, None]

        vel = data["vel"][None, ::self.interval, :, :].astype(np.float32)        # [1, T, n, 2]
        stress = np.concatenate((data["Sigma_xy"][None, ::self.interval, :, None],
                                 data["Sigma_xx"][None, ::self.interval, :, None],
                                 data["Sigma_yy"][None, ::self.interval, :, None]), axis=-1)   # [1, T, n, 3]
        if stress.ndim == 5 and stress.shape[3] == 1:
            stress = np.squeeze(stress, axis=3)

        for key in ("u_in", "u_0", "u0"):
            if key in data:
                u_in = data[key]
                break
        Re = data["Re"]
        if "Bi" in data:
            second = data["Bi"]
        elif "Wi" in data:
            second = data["Wi"]
        else:
            second = 0

        if self.velocity_scaling == "all":
            u_in = u_in * 100
            vel = vel * 100
        elif "Bi" in data or "Wi" in data:
            vel = 100 * vel
        return vel, stress, node_type, pos, np.array([Re, second, u_in])

    def _prepare(self):
        vel_all, stress_all, node_all, pos_all, var_all = [], [], [], [], []
        for fname in self.fname_lst:
            vel, stress, node_type, pos, variables = self._load(fname)
            vel_all.append(vel)
            stress_all.append(stress)
            node_all.append(node_type[None, ...])
            pos_all.append(pos[None, ...])
            var_all.append(variables)
        vel_all = np.concatenate(vel_all, axis=0)
        stress_all = np.concatenate(stress_all, axis=0)
        pos_all = np.concatenate(pos_all, axis=0)
        node_all = np.concatenate(node_all, axis=0)
        var_all = np.array(var_all)
        print(f"{self.data_path}: {len(self.fname_lst)} files, vel {vel_all.shape}, stress {stress_all.shape}")

        s = {}
        s["u_mean"], s["u_std"] = np.mean(vel_all[..., 0]), np.std(vel_all[..., 0])
        s["v_mean"], s["v_std"] = np.mean(vel_all[..., 1]), np.std(vel_all[..., 1])
        s["sxy_mean"], s["sxy_std"] = np.mean(stress_all[..., 0]), np.std(stress_all[..., 0])
        s["sxx_mean"], s["sxx_std"] = np.mean(stress_all[..., 1]), np.std(stress_all[..., 1])
        s["syy_mean"], s["syy_std"] = np.mean(stress_all[..., 2]), np.std(stress_all[..., 2])
        self.statistics = s

        for i in range(len(self.fname_lst)):
            vel, stress, pos = vel_all[i], stress_all[i], pos_all[i]
            if self.use_normalized:
                vel = np.concatenate(((vel[..., 0] - s["u_mean"])[..., None] / s["u_std"],
                                      (vel[..., 1] - s["v_mean"])[..., None] / s["v_std"]), axis=-1)
                stress = np.concatenate(((stress[..., 0] - s["sxy_mean"])[..., None] / s["sxy_std"],
                                         (stress[..., 1] - s["sxx_mean"])[..., None] / s["sxx_std"],
                                         (stress[..., 2] - s["syy_mean"])[..., None] / s["syy_std"]), axis=-1)
            x = np.concatenate((vel[:self.tw], stress[:self.tw], pos[None, ...].repeat(self.tw, axis=0)), axis=-1)
            y = np.concatenate((vel[self.tw:], stress[self.tw:]), axis=-1)
            self.cache[i] = (x.astype(np.float32), y.astype(np.float32), node_all[i].astype(np.int64), pos, var_all[i])

    def global_stats(self):
        """Per-channel mean and standard deviation over the whole folder, in channel order vx, vy, sxy, sxx, syy."""
        s = self.statistics
        mean = np.array([s["u_mean"], s["v_mean"], s["sxy_mean"], s["sxx_mean"], s["syy_mean"]])
        std = np.array([s["u_std"], s["v_std"], s["sxy_std"], s["sxx_std"], s["syy_std"]])
        return mean, std

    def __getitem__(self, idx):
        return self.cache[idx]


def make_loader(dataset, batch_size, shuffle, num_workers=10):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)
