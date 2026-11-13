"""RheOFormer decoder: queries the encoded input at every node and marches the latent state forward in time.

Adapted from OFormer (``IrregSTDecoder2D``, ``CrossFormer``, ``GaussianFourierFeatureTransform``); parameter names
are unchanged.
"""
import numpy as np
import torch
from einops import rearrange
from torch import nn

from .layers import CrossLinearAttention, FeedForward, LinearAttention


class GaussianFourierFeatures(nn.Module):
    """Random Fourier features of the coordinates (Tancik et al. 2020): [sin(2 pi x B), cos(2 pi x B)]."""

    def __init__(self, num_input_channels, mapping_size, scale):
        super().__init__()
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(rearrange(x, "b n c -> (b n) c") @ self._B.to(x.device), "(b n) c -> b n c", b=b)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    """Cross-attention from query points to the encoded input, then a feed-forward layer, both with residuals."""

    def __init__(self, dim, heads, dim_head, mlp_dim, scale, min_freq):
        super().__init__()
        self.cross_attn_module = CrossLinearAttention(dim, heads=heads, dim_head=dim_head, scale=scale,
                                                      min_freq=min_freq)
        self.ffn = FeedForward(dim, mlp_dim)

    def forward(self, x, z, x_pos, z_pos):
        x = self.cross_attn_module(x, z, x_pos, z_pos) + x
        return self.ffn(x) + x


class Decoder(nn.Module):
    """Latent (b, n, latent) -> prediction (b, steps, n, out_channels).

    Each node is queried with Fourier features of its coordinates and its wall indicator; the resulting latent state is
    advanced by the propagator (an MLP with residual, shared over time) and decoded to the five fields at every step."""

    def __init__(self, max_node_type, latent_channels, out_channels, res=200, scale=8, dropout=0.1):
        super().__init__()
        c = latent_channels
        self.out_channels = out_channels
        self.latent_channels = c
        self.node_type_embedding = nn.Embedding(max_node_type, c)
        self.coordinate_projection = nn.Sequential(GaussianFourierFeatures(2, c // 2, scale=scale),
                                                   nn.Linear(c, c, bias=False), nn.GELU(), nn.Linear(c, c, bias=False))
        self.combine_layer = nn.Linear(c * 2, c, bias=False)
        self.input_dropout = nn.Dropout(dropout)
        self.decoding_transformer = CrossFormer(c, heads=4, dim_head=c, mlp_dim=c, scale=32., min_freq=1 / res)
        self.mix_layer = LinearAttention(c, heads=1, dim_head=c, scale=32, min_freq=1 / res)
        self.expand_layer = nn.Linear(c, c * 2, bias=False)
        self.propagator = nn.ModuleList([nn.ModuleList([
            nn.LayerNorm(c * 2),
            nn.Sequential(nn.Linear(c * 2 + 2, c * 2, bias=False), nn.GELU(),
                          nn.Linear(c * 2, c * 2, bias=False), nn.GELU(),
                          nn.Linear(c * 2, c * 2, bias=False), nn.GELU(),
                          nn.Linear(c * 2, c * 2, bias=False))])])
        self.out_norm = nn.LayerNorm(c * 2)
        self.to_out = nn.Sequential(nn.Linear(c * 2, c * 2, bias=False), nn.ReLU(inplace=True),
                                    nn.Linear(c * 2, c, bias=False), nn.ReLU(inplace=True),
                                    nn.Linear(c, out_channels, bias=True))

    def propagate(self, z, pos):
        for norm, mlp in self.propagator:
            z = mlp(torch.cat((norm(z), pos), dim=-1)) + z
        return z

    def forward(self, z, pos, node_type, steps):
        x = self.coordinate_projection(pos)
        x = self.combine_layer(torch.cat((x, self.node_type_embedding(node_type.squeeze(-1))), dim=-1))
        z = self.decoding_transformer(x, self.input_dropout(z), pos, pos)
        z = self.mix_layer(z, pos) + z
        z = self.expand_layer(z)
        out = []
        for _ in range(steps):
            z = self.propagate(z, pos)
            out.append(self.to_out(self.out_norm(z)))
        return torch.stack(out, dim=1)
