"""RheOFormer encoder: embeds the input window at every mesh node and mixes the nodes with linear attention.

Adapted from OFormer (``IrregSTEncoder2D`` and ``TransformerCatNoCls``); parameter names are unchanged.
"""
import torch
from einops.layers.torch import Rearrange
from torch import nn

from .layers import FeedForward, LinearAttention


class AttentionStack(nn.Module):
    """``depth`` blocks of LayerNorm -> linear self-attention (+ residual) -> LayerNorm -> feed-forward (+ residual).
    ``scales`` sets the rotary-embedding scale of each block."""

    def __init__(self, dim, heads, dim_head, mlp_dim, scales, min_freq):
        super().__init__()
        self.layers = nn.ModuleList()
        for scale in scales:
            attn = LinearAttention(dim, heads=heads, dim_head=dim_head, scale=scale, min_freq=min_freq)
            self.layers.append(nn.ModuleList([nn.LayerNorm(dim), attn, nn.LayerNorm(dim), FeedForward(dim, mlp_dim)]))

    def forward(self, x, pos):
        for ln1, attn, ln2, ffn in self.layers:
            x = ln1(x)
            x = attn(x, pos) + x
            x = ln2(x)
            x = ffn(x) + x
        return x


class Encoder(nn.Module):
    """Input (b, tw, n, c) -> latent (b, n, out_channels).

    The time window is collapsed by a convolution over time, the wall indicator is added as a learned embedding, and
    eight attention blocks mix the nodes."""

    def __init__(self, input_channels, time_window, emb_dim=256, out_channels=256, max_node_type=2, heads=1, depth=8,
                 res=200, emb_dropout=0.0):
        super().__init__()
        self.to_embedding = nn.Sequential(
            Rearrange("b t n c -> b c t n"),
            nn.Conv2d(input_channels, emb_dim, kernel_size=(time_window, 1), stride=(time_window, 1), bias=False),
            nn.GELU(),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=(1, 1), bias=False),
            Rearrange("b c 1 n -> b n c"),
        )
        self.node_embedding = nn.Embedding(max_node_type, emb_dim)
        self.combine_embedding = nn.Linear(emb_dim * 2, emb_dim, bias=False)
        self.dropout = nn.Dropout(emb_dropout)
        scales = [32, 16, 8, 8] + [1] * (depth - 4) if depth > 4 else [32] + [16] * (depth - 2) + [1]
        self.s_transformer = AttentionStack(emb_dim, heads, emb_dim, emb_dim, scales, min_freq=1 / res)
        self.ln = nn.LayerNorm(emb_dim)
        self.to_out = nn.Sequential(nn.Linear(emb_dim, emb_dim, bias=False), nn.ReLU(),
                                    nn.Linear(emb_dim, out_channels, bias=False))

    def forward(self, x, node_type, pos):
        x = self.to_embedding(x)
        x = self.combine_embedding(torch.cat([x, self.node_embedding(node_type.squeeze(-1))], dim=-1))
        x_skip = x
        x = self.s_transformer(self.dropout(x), pos)
        return self.to_out(self.ln(x + x_skip))

