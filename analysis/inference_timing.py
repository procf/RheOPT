"""Inference time of a trained model on its validation cases (median of repeated forward passes).

    python analysis/inference_timing.py configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 \
        --arm finetune [--device cuda:0] [--repeats 20]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rheopt.config import load_config  # noqa: E402
from rheopt.engine import load_dataset, predict  # noqa: E402
from rheopt.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--fluid", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", default="finetune", choices=["finetune", "scratch"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config, a.arm, a.fluid, a.n, a.seed, [f"device={a.device}"])
    model = build_model(cfg, cfg.device)
    state = torch.load(os.path.join(cfg.run_dir, "checkpoint.pth"), map_location=cfg.device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    stats = None
    if cfg.normalization == "global":
        s = np.load(os.path.join(cfg.run_dir, "stats.npz"))
        stats = tuple(torch.tensor(s[k], dtype=torch.float32, device=cfg.device) for k in ("mean", "std"))
    ds = load_dataset(cfg, cfg.valid_path)
    sync = torch.cuda.synchronize if cfg.device.startswith("cuda") else (lambda: None)
    for batch in (1, len(ds)):
        items = [ds[i] for i in range(batch)]
        x, y, nt, pos = (torch.tensor(np.stack([it[k] for it in items])) for k in range(4))
        times = []
        for _ in range(3):             # warm-up
            predict(model, cfg, x.clone(), y, pos, nt, stats)
        for _ in range(a.repeats):
            sync()
            t0 = time.perf_counter()
            predict(model, cfg, x.clone(), y, pos, nt, stats)
            sync()
            times.append(time.perf_counter() - t0)
        ms = 1000 * float(np.median(times))
        print(f"batch {batch}: {ms:.1f} ms per forward pass, {ms / batch:.2f} ms per case "
              f"({pos.shape[1]} nodes, {y.shape[1]} predicted frames, {cfg.device})")


if __name__ == "__main__":
    main()
