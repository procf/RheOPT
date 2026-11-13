#!/bin/bash
# Example SLURM job: one setup, one fluid, all N and seeds. Adapt partition, account and environment.
#SBATCH --job-name=rheopt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source activate rheopt
cd "$SLURM_SUBMIT_DIR"
scripts/run_setup.sh configs/contraction/revin_2s.yaml "${FLUID:-Oldroyd}" "4 8 12 16" "1 2 3 4"
