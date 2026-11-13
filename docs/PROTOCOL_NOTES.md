# Protocol notes

This code is a cleaned version of the scripts that produced the paper's results (one copy of the training code per run
folder, plus separate pretraining scripts). It was checked against them:

* **Data:** for every loader variant (contraction and wake; pretraining, fine-tuning, from-scratch, validation; RevIN and
  Global), `rheopt.data.FlowDataset` returns arrays bit-identical to the original loader, including the global
  statistics and the file order (`local` check `check_equivalence.py`, run on Explorer).
* **Model:** the model code is reduced to the configuration RheOFormer uses (the original was the general OFormer
  library). For a given seed it creates the same parameters, with the same names and initial values, as the original
  `get_backbone` (6,775,951 parameters with two input frames and RevIN, 6,790,277 with ten input frames), and gives
  bit-identical outputs and gradients in training mode (with dropout) and in evaluation mode. Checkpoints of the
  original code load directly.
* **Training and evaluation:** seed-1 runs repeated with this code reproduce the stored results (see the table in
  `docs/REPRODUCING.md`).

The points below are part of the original protocol. The code keeps them, so that the stored results can be
reproduced; change them deliberately if you use the code for new work.

## Differences between the two arms

| | fine-tuned (RheOPT) | from scratch |
|---|---|---|
| initialization | pretrained checkpoint | random |
| AdamW weight decay | 1e-6 | 1e-4 |
| kept weights | lowest validation loss among the checks every `val_every` epochs | final weights |
| Global normalization statistics | Newtonian source data | target training data |
| validation batch size during training | 4 | 8 (logging only) |

Everything else is shared: epochs (501 contraction, 2001 wake), batch size 4, one-cycle learning rate (maximum 1e-4,
30 % warm-up, div factor 1e4, final div factor 1e6), loss, architecture, data and seed.

There is no separate test set: the fine-tuned arm selects its checkpoint on the same validation folder on which both
arms are evaluated.

## Validation interval

`val_every` is 100 epochs, except 50 for Bingham (RevIN, two frames, all N) and Giesekus (RevIN, two frames, N = 4,
8, 12), as in the original run folders (`val_every_by_run` in `configs/contraction/revin_2s.yaml`). Pretraining logged
the validation loss every 50 (contraction, RevIN) or 200 epochs (others); for pretraining it has no effect on the
weights.

## Wall indicator (node type)

The model receives a 0/1 node type. Its definition differed between the original scripts and is set by `wall_mask`:

| runs | `wall_mask` | nodes marked |
|---|---|---|
| contraction fine-tuning and from scratch | `contraction` (upstream wall, downstream wall, step face) | 445 of 5,425 |
| contraction pretraining | `contraction_no_step` (without the step face) | 1,601 of 5,433 (Newtonian mesh) |
| wake, Global normalization, fine-tuning and from scratch | `triangle` (obstacle edges) | 43 of 833 |
| wake, RevIN, fine-tuning and from scratch | `contraction` (contraction formula applied to the wake mesh) | 1 of 833 |
| wake pretraining (both normalizations) | `contraction_no_step` | 1 of 2,304 |

So the wake models were pretrained with (almost) no wall nodes, and the wake RevIN runs also fine-tuned without the
obstacle marked.

## Units

The loader converts velocities to cm/s: non-Newtonian files store m/s (multiplied by 100); the Newtonian contraction
files are already in cm/s; for the wake every file is multiplied by 100 (`velocity_scaling: all`).

## Other details

* In the linear attention layers, keys and values are normalized per token over the head channels. The original code
  applied `nn.InstanceNorm1d(dim_head)` to tensors shaped (batch·heads, nodes, channels), which normalizes over the
  channels of each node, not over the nodes; `rheopt/models/layers.py` does the same operation explicitly.

* Files are read in `glob` order (directory order), as originally; this order decides which simulations share a batch.
  Bit-identical reproduction therefore needs the same directory listing, not only the same files.
* The model uses a curriculum on the prediction horizon during the first 20 % of the epochs (4 frames, growing to the
  full horizon); the validation loss during training is computed at the current curriculum stage. Evaluation always uses
  the full horizon.
* Stresses are total stresses (−p I + τ), so the stress error is dominated by the normal components.
* Evaluation metrics: per validation case, the relative L2 error over all predicted frames and nodes of the velocity
  channels and of the stress channels; tables report the mean over cases.
* The Giesekus, Global normalization, ten-frame, N = 16 fine-tuning runs were labelled `N_finetune: 15` in their
  original configuration and result files; the loader ignores that value and the training log confirms that all 16
  simulations were used.
* The original wake RevIN N = 16 runs of seeds 1–3 (from-scratch arm) and seeds 1 and 3 (fine-tuned arm) read
  their training data from a copy of `finetune_N16` (identical files) in another directory, so their file order, and
  hence their batches, may differ from a rerun on `data/wake/Oldroyd/finetune_N16`.
