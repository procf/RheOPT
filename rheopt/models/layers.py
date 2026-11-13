"""Building blocks of RheOFormer: feed-forward layer, rotary position embedding and linear (Galerkin-type) attention.

Adapted from the OFormer code base (Z. Li, K. Meidani, A. Barati Farimani, TMLR 2023), reduced to the configuration
RheOFormer uses: Galerkin attention, rotary embedding of the 2-D node coordinates, no padding. Parameter names are
unchanged, so checkpoints of the original code load directly.
"""
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.init import orthogonal_


class GeGELU(nn.Module):
    """GELU-gated linear unit: GELU(first half of the channels) * second half."""

    def forward(self, x):
        c = x.shape[-1] // 2
        return F.gelu(x[..., :c]) * x[..., c:]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim * 2), GeGELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class RotaryEmbedding(nn.Module):
    """Rotary frequencies for one coordinate (RoFormer), scaled by ``scale / min_freq``."""

    def __init__(self, dim, min_freq, scale):
        super().__init__()
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer("inv_freq", 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, coordinates, device):
        t = coordinates.to(device).type_as(self.inv_freq) * (self.scale / self.min_freq)
        freqs = torch.einsum("... i , j -> ... i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def _rotate_half(x):
    x1, x2 = rearrange(x, "... (j d) -> ... j d", j=2).unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _rotate(t, freqs):
    return t * freqs.cos() + _rotate_half(t) * freqs.sin()


def rotary_freqs(emb, pos, heads, device):
    """Rotary frequencies of the x and y coordinates, (b, h, n, d/2) each. pos: (b, n, 2)."""
    fx = repeat(emb(pos[..., 0], device), "b n d -> b h n d", h=heads)
    fy = repeat(emb(pos[..., 1], device), "b n d -> b h n d", h=heads)
    return fx, fy


def apply_2d_rotary(t, freqs):
    """Rotate the first half of the head channels with the x frequencies and the second half with y."""
    d = t.shape[-1]
    return torch.cat((_rotate(t[..., :d // 2], freqs[0]), _rotate(t[..., d // 2:], freqs[1])), dim=-1)


def _token_norm(t):
    """Normalize each token's head vector to zero mean and unit variance (b, h, n, d).

    The original code applied ``nn.InstanceNorm1d(dim_head)`` to tensors shaped (b*h, n, d); that layer treats n as the
    channel axis and normalizes over d, which is what this function does with the same kernel."""
    b = t.shape[0]
    return rearrange(F.instance_norm(rearrange(t, "b h n d -> (b h) n d"), eps=1e-5), "(b h) n d -> b h n d", b=b)


def _init_identity_plus_orthogonal(weight, rows, gain):
    """Orthogonal initialization of the given row block plus ``gain`` on its diagonal (OFormer initialization)."""
    orthogonal_(weight[rows, :], gain=gain)
    weight.data[rows, :] += gain * torch.diag(torch.ones(weight.size(-1), dtype=torch.float32))


class LinearAttention(nn.Module):
    """Galerkin-type linear self-attention, softmax-free: out = q (k^T v) / n, with k and v normalized per token and
    rotary embedding of the coordinates on q and k (Cao 2021, "Choose a Transformer: Fourier or Galerkin")."""

    def __init__(self, dim, heads, dim_head, scale, min_freq, dropout=0.):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Identity() if (heads == 1 and dim_head == dim) else nn.Sequential(nn.Linear(inner, dim),
                                                                                            nn.Dropout(dropout))
        for h in range(heads):                    # query blocks
            _init_identity_plus_orthogonal(self.to_qkv.weight, slice(h * dim_head, (h + 1) * dim_head), 1. / dim_head)
        self.emb_module = RotaryEmbedding(dim_head // 2, min_freq=min_freq, scale=scale)

    def forward(self, x, pos):
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_qkv(x).chunk(3, dim=-1))
        k, v = _token_norm(k), _token_norm(v)
        freqs = rotary_freqs(self.emb_module, pos, self.heads, x.device)
        q, k = apply_2d_rotary(q, freqs), apply_2d_rotary(k, freqs)
        out = torch.matmul(q, torch.matmul(k.transpose(-1, -2), v)) * (1. / q.shape[2])
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class CrossLinearAttention(nn.Module):
    """Galerkin-type cross-attention: queries from x (query points), keys and values from z (encoded input)."""

    def __init__(self, dim, heads, dim_head, scale, min_freq, dropout=0.):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Identity() if (heads == 1 and dim_head == dim) else nn.Sequential(nn.Linear(inner, dim),
                                                                                            nn.Dropout(dropout))
        g = 1. / dim_head
        for h in range(heads):                    # key and value blocks
            _init_identity_plus_orthogonal(self.to_kv.weight, slice(h * dim_head, (h + 1) * dim_head), g)
            _init_identity_plus_orthogonal(self.to_kv.weight, slice((heads + h) * dim_head, (heads + h + 1) * dim_head), g)
        for h in range(heads):                    # query blocks
            _init_identity_plus_orthogonal(self.to_q.weight, slice(h * dim_head, (h + 1) * dim_head), g)
        self.emb_module = RotaryEmbedding(dim_head // 2, min_freq=min_freq, scale=scale)

    def forward(self, x, z, x_pos, z_pos):
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.to_kv(z).chunk(2, dim=-1))
        k, v = _token_norm(k), _token_norm(v)
        q = apply_2d_rotary(q, rotary_freqs(self.emb_module, x_pos, self.heads, x.device))
        k = apply_2d_rotary(k, rotary_freqs(self.emb_module, z_pos, self.heads, z.device))
        out = torch.matmul(q, torch.matmul(k.transpose(-1, -2), v)) * (1. / z.shape[1])
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))
