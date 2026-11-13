#!/bin/bash
# Run both arms (from scratch, fine-tuned) and evaluate them for every fluid, N and seed of one setup.
# Usage: scripts/run_setup.sh <config> "<fluids>" "<N values>" "<seeds>"
#   scripts/run_setup.sh configs/contraction/revin_2s.yaml "Giesekus Oldroyd Bingham" "4 8 12 16" "1 2 3 4"
#   scripts/run_setup.sh configs/wake/global.yaml "Oldroyd" "4 8 12 16 24 32" "1 2 3 4"
# Each run takes 2-5 min (contraction) or 10-30 min (wake) on one H200. On a SLURM cluster, wrap one call per job
# (see scripts/slurm_example.sbatch).
set -euo pipefail
cfg=$1; fluids=$2; ns=$3; seeds=$4
for f in $fluids; do for n in $ns; do for s in $seeds; do for arm in scratch finetune; do
  python -m rheopt "$arm" "$cfg" --fluid "$f" --n "$n" --seed "$s" --evaluate
done; done; done; done
