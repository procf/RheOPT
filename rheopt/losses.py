"""Training loss and evaluation metrics."""
import numpy as np
import torch


def relative_l2_loss(pred, target, eps=1e-6):
    """Training loss: relative L2 error over (time, nodes) per sample and channel, summed over channels and samples,
    divided by the batch size."""
    diff = torch.norm(pred - target, p=2, dim=(1, 2)) / (torch.norm(target, p=2, dim=(1, 2)) + eps)
    return torch.sum(diff) / pred.shape[0]


def relative_l2(pred, target, eps=1e-8):
    """Reported metric: ||pred - target|| / ||target|| over all predicted frames, nodes and the given channels."""
    return (torch.norm(pred - target) / (torch.norm(target) + eps)).item()


def nrmse(pred, target, eps=1e-8):
    """RMSE divided by the standard deviation of the target."""
    return (torch.sqrt(torch.mean((pred - target) ** 2)) / (torch.std(target) + eps)).item()


def per_sample_metrics(pred, target, n_vel=2):
    """Per-sample relative L2 and NRMSE for velocity (channels 0-1), stress (2-4) and all channels."""
    out = {k: [] for k in ("vel_rel_l2", "vel_nrmse", "stress_rel_l2", "stress_nrmse", "all_rel_l2", "all_nrmse")}
    for p, t in zip(pred, target):
        for name, sl in (("vel", slice(0, n_vel)), ("stress", slice(n_vel, None)), ("all", slice(None))):
            out[f"{name}_rel_l2"].append(relative_l2(p[..., sl], t[..., sl]))
            out[f"{name}_nrmse"].append(nrmse(p[..., sl], t[..., sl]))
    return out


def summary(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)),
            "max": float(np.max(values)), "sum": float(np.sum(values))}
