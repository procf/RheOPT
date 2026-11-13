"""Command line: python -m rheopt {pretrain,scratch,finetune,evaluate} CONFIG [options]

Examples
    python -m rheopt pretrain configs/pretrain/contraction_revin_2s.yaml
    python -m rheopt finetune configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
    python -m rheopt scratch  configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
    python -m rheopt evaluate configs/contraction/revin_2s.yaml --arm finetune --fluid Oldroyd --n 4 --seed 1
"""
import argparse
import os

from .config import load_config, save_config


def main():
    ap = argparse.ArgumentParser(prog="python -m rheopt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pretrain", "scratch", "finetune", "evaluate"])
    ap.add_argument("config")
    ap.add_argument("--fluid", help="Giesekus, Oldroyd or Bingham")
    ap.add_argument("--n", type=int, help="number of fine-tuning simulations")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--arm", choices=["scratch", "finetune"], help="for evaluate: which arm's checkpoint")
    ap.add_argument("--evaluate", action="store_true", help="evaluate after training")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args()

    from . import engine  # imported here so --help works without torch

    arm = a.arm if a.command == "evaluate" else a.command
    if arm is None:
        ap.error("evaluate needs --arm")
    cfg = load_config(a.config, arm, a.fluid, a.n, a.seed, a.set)
    if a.command == "evaluate":
        engine.evaluate(cfg)
        return
    os.makedirs(cfg.run_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.run_dir, "config.yaml"))
    print(f"{a.command}: {cfg.config_file} fluid={cfg.fluid} N={cfg.n} seed={cfg.seed} -> {cfg.run_dir}")
    if a.command == "pretrain":
        engine.train_from_scratch(cfg, pretrain=True)
    elif a.command == "scratch":
        engine.train_from_scratch(cfg)
    else:
        engine.finetune(cfg)
    if a.evaluate and a.command != "pretrain":
        engine.evaluate(cfg)


if __name__ == "__main__":
    main()
