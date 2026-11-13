# Reproducing the results

All commands run from the repository root, with the data in `data/` (or `$RHEOPT_DATA`) and outputs in `runs/`
(or `$RHEOPT_RUNS`). Each run writes `config.yaml`, `train_log.csv`, `checkpoint.pth`, `metrics.json` and
`predictions.npz` to its run folder.

## 1. Pretraining on Newtonian flow

| config | used for | time on one H200 |
|---|---|---|
| `configs/pretrain/contraction_revin_2s.yaml` | contraction, RevIN, two input frames (main results) | ≈ 1.9 h |
| `configs/pretrain/contraction_global_2s.yaml` | contraction, Global normalization, two frames | ≈ 1.9 h |
| `configs/pretrain/contraction_revin_10s.yaml` | contraction, RevIN, ten frames (SI) | ≈ 1.9 h |
| `configs/pretrain/contraction_global_10s.yaml` | contraction, Global, ten frames (SI) | ≈ 1.9 h |
| `configs/pretrain/wake_global.yaml` | wake, Global normalization (main results) | ≈ 2.7 h |
| `configs/pretrain/wake_revin.yaml` | wake, RevIN | ≈ 2.7 h |

```bash
python -m rheopt pretrain configs/pretrain/contraction_revin_2s.yaml
```

The checkpoint is written to `runs/pretrain/<name>/pretrained.pth`, where the fine-tuning configs look for it. The
pretrained checkpoints of the paper: **[[TODO: checkpoint download link]]**.

## 2. Fine-tuning and training from scratch

| config | fluids | N | paper |
|---|---|---|---|
| `configs/contraction/revin_2s.yaml` | Giesekus, Oldroyd, Bingham | 4, 8, 12, 16 | main contraction results |
| `configs/contraction/global_2s.yaml` | same | same | normalization comparison |
| `configs/contraction/revin_10s.yaml`, `global_10s.yaml` | same | same | SI, ten-frame input window |
| `configs/wake/global.yaml` | Oldroyd | 4, 8, 12, 16, 24, 32 | main wake results |
| `configs/wake/revin.yaml` | Oldroyd | same | wake normalization comparison |

One run:

```bash
python -m rheopt finetune configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
python -m rheopt scratch  configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
```

A whole setup (seeds 1–4 as in the paper): `scripts/run_setup.sh configs/contraction/revin_2s.yaml "Giesekus Oldroyd Bingham" "4 8 12 16" "1 2 3 4"`.

Any config value can be overridden with `--set key=value` (for example `--set epochs=50 num_workers=2` for a quick
test).

### Reproduction check

Seed-1 runs repeated with this code on Explorer (NVIDIA H200, torch 2.5.1, CUDA 12.1), compared with the stored
results of the original code:

| setup | fluid, N | arm | velocity rel. L2 (new / stored) | stress rel. L2 (new / stored) | per case |
|---|---|---|---|---|---|
| contraction RevIN, 2 frames | Oldroyd-B, 4 | from scratch | 0.046779 / 0.046779 | 0.196647 / 0.196647 | identical |
| contraction RevIN, 2 frames | Oldroyd-B, 4 | fine-tuned | 0.029481 / 0.029481 | 0.037557 / 0.037557 | identical |
| contraction Global, 2 frames | Oldroyd-B, 4 | from scratch | 0.248936 / 0.248936 | 0.243115 / 0.243115 | identical |
| contraction Global, 2 frames | Oldroyd-B, 4 | fine-tuned | 0.167146 / 0.167146 | 0.314426 / 0.314426 | identical |
| contraction RevIN, 2 frames | Bingham, 4 | fine-tuned (validation every 50 epochs) | 0.039655 / 0.039655 | 0.089615 / 0.089615 | identical |
| wake Global | Oldroyd-B, 4 | from scratch | 0.125274 / 0.125288 | 0.185508 / 0.185516 | within 0.09 % |
| wake Global | Oldroyd-B, 4 | fine-tuned | 0.085939 / 0.085964 | 0.174133 / 0.174154 | within 0.22 % |

(all cases of the validation folder; seed 1)

Further checks against the original code and results (run on Explorer):

* data loading for every loader variant, model initialization, and model outputs and gradients in training and
  evaluation mode: bit-identical;
* `analysis/wake_frequency.py`, `analysis/stress_components.py` and `analysis/persistence_baseline.py` on the
  original runs (seeds 1–3) reproduce every value of the corresponding tables of the revision.

Training is deterministic for the contraction on this hardware. The wake runs differ slightly from run to run (the
same size of difference occurs between two runs of the original code); results may also differ on other GPUs or
library versions.

## 3. Additional experiments (revision)

These use options of the same code.

| experiment | how |
|---|---|
| E2: pretraining-set size | `python scripts/make_newtonian_subsets.py`, then `python -m rheopt pretrain configs/pretrain/contraction_revin_2s.yaml --set train_path=data/contraction/Newtonian_subsets/Npre25 epochs=3900 run_dir=runs/pretrain/Npre25` (epochs scaled to the same number of optimizer steps: 3900, 1950, 1114 and 600 for N_pre = 25, 50, 100, 200; second pretraining seed: `seed=2`), then fine-tune with `--set pretrained_path=runs/pretrain/Npre25/pretrained.pth run_dir=...` |
| E3: cross-geometry transfer | fine-tune a wake config from a contraction checkpoint with the same input window: `--set pretrained_path=runs/pretrain/contraction_revin_10s/pretrained.pth`; under Global normalization add `stats_override=<npz with mean, std of the source data>`. Reverse direction: pretrain on the wake data with the two-frame contraction settings (`configs/pretrain/contraction_revin_2s.yaml --set train_path=data/wake/Newtonian valid_path=data/wake/Newtonian velocity_scaling=all`) and fine-tune the contraction from it |
| E4: joint training | `python -m rheopt scratch configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate --set joint_newtonian_path=data/contraction/Newtonian joint_newtonian_batch_size=16 joint_newtonian_passes=600 joint_target_passes=501 epochs=600` (same batches as pretraining plus fine-tuning, interleaved) |
| E5: frozen components | `python -m rheopt finetune ... --set freeze=encoder` (or `propagator`, `decoder`) |
| D-035: final instead of best checkpoint | `python -m rheopt finetune ... --set save_final=true` (the best checkpoint is kept as `checkpoint_best.pth`) |

## 4. Tables and analyses

* `python analysis/summarize_runs.py` collects the `metrics.json` of all runs into one table.
* `analysis/`: wake frequency, stress components, persistence baseline, field statistics, timing
  (`analysis/README.md`). Outputs go to `results/` (not tracked).
