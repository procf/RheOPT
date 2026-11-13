"""Field normalization: RevIN (per sample) and Global (per dataset).

RevIN (Kim et al., ICLR 2022) normalizes each sample and channel with the mean and standard deviation of its own input
window, applies a learnable per-channel affine map, and the targets are normalized with the same input statistics
(without the affine map). Predictions are mapped back with ``pred * std + mean``.

Global normalization uses one mean and standard deviation per channel computed over a whole training folder: the
target folder when training from scratch, the Newtonian source folder when fine-tuning a pretrained model.
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization over (time, node) for tensors shaped (B, T, N, C)."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, num_channels))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_channels))

    def normalize(self, x):
        if x.dim() != 4:
            raise ValueError(f"expected (B, T, N, C), got {tuple(x.shape)}")
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + self.eps
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, {"mean": mean, "std": std}


def revin_inputs(revin, x, y):
    """Normalize a batch: the five field channels of ``x`` in place with RevIN, ``y`` with the statistics of the input
    window (no affine map). Returns (x, y_norm, stats); predictions map back with ``pred * std + mean``."""
    x[..., 0:5], stats = revin.normalize(x[..., 0:5])
    y_norm = (y - stats["mean"]) / stats["std"]
    return x, y_norm, stats
