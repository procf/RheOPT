"""Training and evaluation.

* ``train_from_scratch``  random initialization; used for Newtonian pretraining and for the from-scratch baseline.
                          AdamW weight decay 1e-4 (``arms`` in configs/base.yaml); keeps the final weights.
* ``finetune``            initialization from a pretrained checkpoint; weight decay 1e-6; keeps the weights with the
                          lowest validation loss among the checks every ``val_every`` epochs.
* ``evaluate``            scores a checkpoint on ``valid_path`` and writes metrics.json and predictions.npz.

Both training paths use the one-cycle learning-rate schedule (maximum ``lr``, 30 % warm-up, div factor 1e4, final div
factor 1e6, one step per batch) and the relative L2 loss. The order of operations and of random-number draws follows
the original scripts, so runs reproduce the stored results (docs/PROTOCOL_NOTES.md).
"""
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .data import FlowDataset, make_loader
from .losses import per_sample_metrics, relative_l2_loss, summary
from .models import build_model, load_pretrained, save_pretrained
from .normalization import revin_inputs


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(cfg, path, use_normalized=False):
    """FlowDataset of one folder, with the loader options of the config."""
    return FlowDataset(path, cfg.time_window, wall_mask=cfg.wall_mask, velocity_scaling=cfg.velocity_scaling,
                       use_normalized=use_normalized)


def _optimizer_and_schedule(model, cfg, steps_per_epoch):
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
                                                    steps_per_epoch=steps_per_epoch, div_factor=1e4,
                                                    pct_start=0.3, final_div_factor=1e6)
    return optimizer, scheduler


# ── loss of one batch, training loop, validation ─────────────────────────────

def batch_loss(model, batch, cfg, epoch, stats=None):
    """Relative L2 loss of one batch in normalized space.

    RevIN normalizes the batch with its own input statistics. Global normalization uses ``stats`` = (mean, std) per
    channel; ``stats=None`` means the dataset already returns normalized fields (Global, training from scratch)."""
    x, y, node_type, pos, _ = batch
    device = cfg.device
    if cfg.normalization == "revin":
        x, y, _ = revin_inputs(model.revin, x.to(device), y.to(device))
    elif stats is not None:
        x[..., 0:5] = (x[..., 0:5] - stats[0]) / stats[1]
        y = (y - stats[0]) / stats[1]
    y = y.to(device)
    pred = model(x.to(device), pos.to(device), node_type.to(device), y.shape[1], epoch, cfg.epochs)
    return relative_l2_loss(pred, y[:, :pred.shape[1]])


def train_epoch(model, optimizer, scheduler, loader, cfg, epoch, stats=None):
    model.train()
    losses = []
    for batch in loader:
        loss = batch_loss(model, batch, cfg, epoch, stats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach() / batch[0].shape[0])
    return torch.mean(torch.stack(losses)).item()


@torch.no_grad()
def validation_loss(model, loader, cfg, epoch, stats=None):
    """Validation loss as used for checkpoint selection (normalized space, curriculum stage of ``epoch``)."""
    model.eval()
    losses = [batch_loss(model, batch, cfg, epoch, stats) / batch[0].shape[0] for batch in loader]
    return torch.mean(torch.stack(losses)).item()


def _fit(model, optimizer, scheduler, train_loader, valid_loader, cfg, train_stats, valid_stats, on_validation=None,
         sampler=None):
    """Train for ``cfg.epochs`` epochs, compute the validation loss every ``cfg.val_every`` epochs (starting at epoch
    0), write train_log.csv and call ``on_validation(epoch, val_loss)`` after each validation."""
    log_path = os.path.join(cfg.run_dir, "train_log.csv")
    with open(log_path, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(cfg.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, cfg, epoch, train_stats)
            val_loss = ""
            if epoch % cfg.val_every == 0:
                val_loss = validation_loss(model, valid_loader, cfg, epoch, valid_stats)
                print(f"epoch {epoch}: train {train_loss:.5f} validation {val_loss:.5f}", flush=True)
                if on_validation:
                    on_validation(epoch, val_loss)
            log.writerow([epoch, train_loss, val_loss])
            fh.flush()


# ── E4 joint training: Newtonian and target batches interleaved ──────────────

def _shuffled_batches(n, batch_size, offset, passes):
    out = []
    for _ in range(passes):
        perm = (torch.randperm(n) + offset).tolist()
        out += [perm[i:i + batch_size] for i in range(0, n, batch_size)]
    return out


class JointBatchSampler:
    """Exactly the batches of pretraining (``newt_passes`` passes over the Newtonian set, batches of ``b_newt``) and of
    fine-tuning (``target_passes`` passes over the target set, batches of ``b_target``), shuffled together once and
    spread evenly over the epochs. Every batch comes from one source (the meshes differ). Target samples come first
    in the concatenated dataset."""

    def __init__(self, n_target, n_newt, b_target, b_newt, target_passes, newt_passes, epochs):
        batches = (_shuffled_batches(n_newt, b_newt, n_target, newt_passes)
                   + _shuffled_batches(n_target, b_target, 0, target_passes))
        self.schedule = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        self.bounds = [len(self.schedule) * e // epochs for e in range(epochs + 1)]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        e = min(max(self.epoch, 0), len(self.bounds) - 2)
        return iter(self.schedule[self.bounds[e]:self.bounds[e + 1]])

    def __len__(self):
        return max(b - a for a, b in zip(self.bounds[:-1], self.bounds[1:]))


# ── training from scratch / pretraining ──────────────────────────────────────

def train_from_scratch(cfg, pretrain=False):
    """Random initialization, final weights kept. ``pretrain=True`` writes ``pretrained.pth`` (encoder, decoder and
    full state dict), the format ``finetune`` loads; otherwise ``checkpoint.pth``."""
    os.makedirs(cfg.run_dir, exist_ok=True)
    seed_everything(cfg.seed)

    train_set = load_dataset(cfg, cfg.train_path, use_normalized=cfg.normalization == "global")
    stats = train_set.global_stats()
    train_loader = make_loader(train_set, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    sampler = None
    if getattr(cfg, "joint_newtonian_path", None):
        if cfg.normalization != "revin":
            raise ValueError("joint training is implemented for RevIN only")
        newt_set = load_dataset(cfg, cfg.joint_newtonian_path)
        sampler = JointBatchSampler(len(train_set), len(newt_set), cfg.batch_size, cfg.joint_newtonian_batch_size,
                                    cfg.joint_target_passes, cfg.joint_newtonian_passes, cfg.epochs)
        train_loader = DataLoader(ConcatDataset([train_set, newt_set]), batch_sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=True,
                                  persistent_workers=cfg.num_workers > 0)
        print(f"joint training: {len(train_set)} target + {len(newt_set)} Newtonian samples, "
              f"{len(sampler.schedule)} optimizer steps")
    valid_loader = make_loader(load_dataset(cfg, cfg.valid_path), cfg.valid_batch_size, shuffle=False,
                               num_workers=cfg.num_workers)

    model = build_model(cfg, cfg.device)
    optimizer, scheduler = _optimizer_and_schedule(model, cfg, len(train_loader))
    if cfg.normalization == "global":
        np.savez(os.path.join(cfg.run_dir, "stats.npz"), mean=stats[0], std=stats[1])

    _fit(model, optimizer, scheduler, train_loader, valid_loader, cfg, train_stats=None, valid_stats=stats,
         sampler=sampler)
    if pretrain:
        save_pretrained(model, os.path.join(cfg.run_dir, "pretrained.pth"))
    else:
        torch.save({"epoch": cfg.epochs - 1, "model_state_dict": model.state_dict()},
                   os.path.join(cfg.run_dir, "checkpoint.pth"))
    return model


# ── fine-tuning ──────────────────────────────────────────────────────────────

def _freeze(model, parts):
    """E5: keep pretrained components fixed: ``encoder``, ``propagator`` (latent time-marching MLP) and/or
    ``decoder`` (all other decoder parameters). Comma-separated."""
    for part in [p for p in str(parts or "").split(",") if p]:
        if part == "encoder":
            frozen = list(model.encoder.parameters())
        elif part == "propagator":
            frozen = list(model.decoder.propagator.parameters())
        elif part == "decoder":
            frozen = [p for n, p in model.decoder.named_parameters() if not n.startswith("propagator.")]
        else:
            raise ValueError(f"unknown freeze target {part}")
        for p in frozen:
            p.requires_grad_(False)
    if parts:
        print("frozen:", parts, "- trainable parameters:",
              sum(p.numel() for p in model.parameters() if p.requires_grad), "of",
              sum(p.numel() for p in model.parameters()))


def finetune(cfg):
    """Initialize from ``cfg.pretrained_path`` and train on the target data, keeping the weights with the lowest
    validation loss. Global normalization uses the statistics of the source data (``source_path``), or those in
    ``stats_override`` (E3, transfer between geometries)."""
    os.makedirs(cfg.run_dir, exist_ok=True)
    stats = None
    if cfg.normalization == "global":
        if getattr(cfg, "stats_override", None):
            s = np.load(cfg.stats_override)
            stats = (s["mean"], s["std"])
        else:
            stats = load_dataset(cfg, cfg.source_path).global_stats()
        np.savez(os.path.join(cfg.run_dir, "stats.npz"), mean=stats[0], std=stats[1])
    seed_everything(cfg.seed)

    train_loader = make_loader(load_dataset(cfg, cfg.train_path), cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers)
    valid_loader = make_loader(load_dataset(cfg, cfg.valid_path), cfg.valid_batch_size, shuffle=False,
                               num_workers=cfg.num_workers)
    model = build_model(cfg, cfg.device)
    optimizer, scheduler = _optimizer_and_schedule(model, cfg, len(train_loader))
    load_pretrained(model, cfg.pretrained_path)
    _freeze(model, getattr(cfg, "freeze", None))

    ckpt_path = os.path.join(cfg.run_dir, "checkpoint.pth")
    best = {"loss": float("inf")}

    def keep_best(epoch, val_loss):
        if val_loss < best["loss"]:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, ckpt_path)
            best["loss"] = val_loss

    _fit(model, optimizer, scheduler, train_loader, valid_loader, cfg, train_stats=stats, valid_stats=stats,
         on_validation=keep_best)
    if getattr(cfg, "save_final", False):          # D-035: score the final weights instead of the best ones
        os.replace(ckpt_path, os.path.join(cfg.run_dir, "checkpoint_best.pth"))
        torch.save({"epoch": cfg.epochs - 1, "model_state_dict": model.state_dict()}, ckpt_path)
    return model


# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, cfg, x, y, pos, node_type, stats=None):
    """Prediction and ground truth in physical units over the full horizon. ``stats`` = (mean, std) tensors on the
    device for Global normalization. (The Global ground truth is round-tripped through the normalization, as in the
    original evaluation.)"""
    device = cfg.device
    if cfg.normalization == "revin":
        gt = y.to(device)
        x, _, st = revin_inputs(model.revin, x.to(device), gt)
        pred = model(x, pos.to(device), node_type.to(device), gt.shape[1], 1, 1)
        return pred * st["std"] + st["mean"], gt
    mean, std = stats
    x[..., 0:5] = (x[..., 0:5].to(device) - mean) / std
    y = (y.to(device) - mean) / std
    pred = model(x.to(device), pos.to(device), node_type.to(device), y.shape[1], 1, 1)
    return pred * std + mean, y * std + mean


@torch.no_grad()
def evaluate(cfg, checkpoint=None, out_dir=None):
    """Score a checkpoint on ``cfg.valid_path``.

    Writes ``metrics.json`` (mean, s.d., min, max of the per-case relative L2 error and NRMSE of velocity, stress and
    all channels, plus per-case values and file names) and ``predictions.npz`` (pos, gt, pred, variables, files)."""
    out_dir = out_dir or cfg.run_dir
    checkpoint = checkpoint or os.path.join(cfg.run_dir, "checkpoint.pth")
    valid_set = load_dataset(cfg, cfg.valid_path)
    loader = make_loader(valid_set, 500, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg, cfg.device)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True)["model_state_dict"])
    model.eval()
    stats = None
    if cfg.normalization == "global":
        s = np.load(os.path.join(os.path.dirname(checkpoint), "stats.npz"))
        stats = tuple(torch.tensor(s[k], dtype=torch.float32, device=cfg.device) for k in ("mean", "std"))

    preds, gts, poss, variables = [], [], [], []
    for x, y, node_type, pos, var in loader:
        pred, gt = predict(model, cfg, x, y, pos, node_type, stats)
        preds.append(pred)
        gts.append(gt)
        poss.append(pos)
        variables.append(var)
    pred, gt = torch.cat(preds), torch.cat(gts)

    m = per_sample_metrics(pred, gt)
    files = [os.path.basename(f) for f in valid_set.fname_lst]
    metrics = {
        "checkpoint": checkpoint, "valid_path": cfg.valid_path, "n_val_samples": int(pred.shape[0]),
        "velocity": summary(m["vel_rel_l2"]), "stress": summary(m["stress_rel_l2"]),
        "combined": summary(m["all_rel_l2"]),
        "velocity_nrmse": summary(m["vel_nrmse"]), "stress_nrmse": summary(m["stress_nrmse"]),
        "combined_nrmse": summary(m["all_nrmse"]),
        "per_sample": {"file": files, "vel_rel_l2": m["vel_rel_l2"], "vel_nrmse": m["vel_nrmse"],
                       "stress_rel_l2": m["stress_rel_l2"], "stress_nrmse": m["stress_nrmse"]},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    if getattr(cfg, "save_predictions", True):
        np.savez(os.path.join(out_dir, "predictions.npz"), pos=torch.cat(poss).numpy(), gt=gt.cpu().numpy(),
                 pred=pred.cpu().numpy(), variables=torch.cat(variables).numpy(), files=np.array(files))
    print(f"velocity rel. L2 {metrics['velocity']['mean']:.4f}, stress rel. L2 {metrics['stress']['mean']:.4f} "
          f"({metrics['n_val_samples']} cases) -> {out_dir}/metrics.json")
    return metrics
