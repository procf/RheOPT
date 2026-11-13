"""Experiment configuration.

A config file describes one experimental setup (geometry, normalization, input window). The fluid, the number of
fine-tuning simulations N, the seed and the arm are given on the command line; ``{placeholders}`` in paths are filled
from them and from ``data_root`` / ``runs_root`` (environment variables RHEOPT_DATA and RHEOPT_RUNS, defaults ``data``
and ``runs``). Values under ``arms.<arm>`` override the top level for that arm; ``val_every_by_run`` maps
"<fluid>" or "<fluid>_N<n>" to a validation interval that differs from ``val_every``.
"""
import os
from types import SimpleNamespace

import yaml

ARMS = ("pretrain", "scratch", "finetune")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "configs", "base.yaml")


def _parse_value(v):
    return yaml.safe_load(v)


def load_config(path, arm, fluid=None, n=None, seed=None, overrides=()):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    with open(BASE) as fh:
        cfg = yaml.safe_load(fh)
    with open(path) as fh:
        exp = yaml.safe_load(fh)
    arms = {**cfg.pop("arms", {}).get(arm, {}), **exp.pop("arms", {}).get(arm, {})}
    cfg.update(exp)
    cfg.update(arms)
    for item in overrides:
        key, _, value = item.partition("=")
        cfg[key] = _parse_value(value)

    cfg["arm"] = arm
    cfg["fluid"] = fluid if fluid is not None else cfg.get("fluid")
    cfg["n"] = n if n is not None else cfg.get("n")
    cfg["seed"] = int(seed if seed is not None else cfg.get("seed", 1))
    cfg.setdefault("data_root", os.environ.get("RHEOPT_DATA", "data"))
    cfg.setdefault("runs_root", os.environ.get("RHEOPT_RUNS", "runs"))
    cfg["config_file"] = os.path.relpath(path)

    by_run = cfg.pop("val_every_by_run", {}) or {}
    for key in (f"{cfg['fluid']}_N{cfg['n']}", f"{cfg['fluid']}"):
        if key in by_run:
            cfg["val_every"] = by_run[key]
            break

    fill = {k: cfg[k] for k in ("fluid", "n", "seed", "arm", "data_root", "runs_root")}
    for key, value in list(cfg.items()):
        if isinstance(value, str) and "{" in value:
            cfg[key] = value.format(**fill)
    return SimpleNamespace(**cfg)


def save_config(cfg, path):
    with open(path, "w") as fh:
        yaml.safe_dump(vars(cfg), fh, sort_keys=True)
