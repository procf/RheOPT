# RheOPT: few-shot non-Newtonian flow prediction by Newtonian pretraining of RheOFormer

Code for

> M. Saberi, A. Barati Farimani, S. Jamali. *RheOPT: Few-Shot Non-Newtonian Flow Prediction via Newtonian
> Pretraining of RheOFormer.* https://arxiv.org/pdf/2610.31465

RheOPT pretrains the RheOFormer neural operator on simulations of Newtonian flow and fine-tunes it on a few
(N = 4–32) simulations of a non-Newtonian fluid in the same geometry. The model predicts the velocity and the stress
fields together on the unstructured simulation mesh. Two geometries are studied: a 4:1 planar contraction (Giesekus,
Oldroyd-B and Bingham fluids) and the wake behind a triangular obstacle (Oldroyd-B).

## Repository layout

```
rheopt/            the package
  data.py            loader of the simulation files (FlowDataset)
  models/            RheOFormer: layers.py (attention), encoder.py, decoder.py, rheoformer.py
  normalization.py   RevIN
  engine.py          pretraining, fine-tuning, training from scratch, evaluation
  config.py          config files and command-line overrides; __main__.py: python -m rheopt
  runs.py, wake.py   locating run outputs; wake frequency and RMS diagnostics (used by analysis/)
configs/           one config per experimental setup (pretraining, contraction, wake); the list of clean wake validation cases
scripts/           run a whole setup, make the Newtonian subsets, example SLURM job
analysis/          scripts for the numbers beyond the mean errors (wake frequency, stress components, ...)
docs/              REPRODUCING.md (commands, checks), PROTOCOL_NOTES.md (implementation details)
data/              not tracked; see data/README.md
```

## Installation

```bash
conda env create -f environment.yml
conda activate rheopt
```

The runs of the paper used Python 3.11, PyTorch 2.5.1 with CUDA 12.1 and one NVIDIA H200 GPU per run.

## Quick start

Put the data in `data/` (see `data/README.md`), then:

```bash
# 1. pretrain on the 200 Newtonian contraction simulations (about 2 h on one H200)
python -m rheopt pretrain configs/pretrain/contraction_revin_2s.yaml

# 2. fine-tune on 4 Oldroyd-B simulations, and train the same model from scratch for comparison
python -m rheopt finetune configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
python -m rheopt scratch  configs/contraction/revin_2s.yaml --fluid Oldroyd --n 4 --seed 1 --evaluate
```

Each command writes its checkpoint, training log, metrics (`metrics.json`) and predictions (`predictions.npz`) to
`runs/<setup>/<fluid>/N<n>/<arm>_seed<seed>/`. See `docs/REPRODUCING.md` for all setups, the additional experiments
and the expected run times.

## Method in brief

* **Model.** RheOFormer: an OFormer encoder on the mesh nodes (input: two or ten frames of vx, vy, σxy, σxx, σyy and the
  node coordinates, plus a wall indicator), a cross-attention decoder and a latent propagator that marches the
  prediction forward in time.
* **Normalization.** RevIN normalizes each sample with the statistics of its own input window (with a learnable
  affine map); Global normalization uses one mean and standard deviation per channel over a training folder.
* **Training.** AdamW, one-cycle learning rate with maximum 1e-4, batch size 4, 501 epochs (contraction) or 2001
  epochs (wake), relative L2 loss. Pretraining: 600 (contraction) or 2000 (wake) epochs, batch size 16.
* **Evaluation.** Relative L2 error per validation case over the full predicted horizon, for the velocity and the
  stress channels, averaged over cases; seeds 1–4.

The fine-tuned and from-scratch arms follow the original protocol, which differs in more than the initialization
(weight decay, checkpoint selection); see `docs/PROTOCOL_NOTES.md`.

## Results

Every run writes `metrics.json` (per-case relative L2 error and NRMSE of velocity and stress) to its run folder;
`python analysis/summarize_runs.py` collects them into one table. The scripts in `analysis/` compute the further
numbers of the paper (wake shedding frequency, stress components, baselines).

## Citation

@article{saberi2026rheopt,
  title={RheOPT: Few-Shot Non-Newtonian Flow Prediction via Newtonian Pretraining of RheOFormer},
  author={Saberi, Maedeh and Farimani, Amir Barati and Jamali, Safa},
  journal={arXiv preprint arXiv:2610.31465},
  year={2026}
}
