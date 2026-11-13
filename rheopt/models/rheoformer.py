"""RheOFormer: encoder, latent propagator and decoder on unstructured meshes, predicting velocity and stress together."""
import torch
from torch import nn

from ..normalization import RevIN
from .decoder import Decoder
from .encoder import Encoder

LATENT = 256          # width of all hidden layers
RES = 200             # sets the lowest rotary-embedding frequency (1 / RES)


def curriculum_steps(n_steps, epoch, num_epochs, start=4, ratio=0.2):
    """Number of frames to predict at ``epoch``: 4 frames at first, growing to all ``n_steps`` frames over the first
    ``ratio`` of the epochs (OFormer curriculum). Evaluation uses epoch = num_epochs = 1, i.e. all frames."""
    if epoch < int(ratio * num_epochs):
        progress = (epoch * 2) / (num_epochs * ratio)
        return start + int(max(0, progress - 1.) * ((n_steps - start) / 2.)) * 2
    return n_steps


class RheOFormer(nn.Module):
    """Input window x (b, tw, n, 7) -> prediction (b, steps, n, 5)."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, pos, node_type, n_steps, epoch, num_epochs):
        z = self.encoder(x, node_type, pos)
        return self.decoder(z, pos, node_type, curriculum_steps(n_steps, epoch, num_epochs))


def build_model(cfg, device):
    """RheOFormer as used in all experiments (6.78 M parameters).

    With RevIN normalization the model also holds the RevIN affine parameters as ``model.revin``; they are created
    after the network, so the network initialization does not depend on the normalization."""
    encoder = Encoder(input_channels=cfg.in_channels, time_window=cfg.time_window, emb_dim=LATENT,
                      out_channels=LATENT, res=RES).to(device)
    decoder = Decoder(max_node_type=2, latent_channels=LATENT, out_channels=cfg.out_channels, res=RES, scale=2,
                      dropout=0.2).to(device)
    model = RheOFormer(encoder, decoder).to(device)
    if cfg.normalization == "revin":
        model.revin = RevIN(num_channels=cfg.out_channels).to(device)
    return model


def load_pretrained(model, path):
    """Load a pretrained checkpoint: encoder and decoder strictly, then the full state dict (this adds the RevIN
    affine parameters when present)."""
    state = torch.load(path, weights_only=True)
    model.encoder.load_state_dict(state["encoder_state_dict"], strict=True)
    model.decoder.load_state_dict(state["decoder_state_dict"], strict=True)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        print("load_pretrained: missing keys", missing, "unexpected keys", unexpected)
    return model


def save_pretrained(model, path):
    torch.save({"model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict()}, path)
