#!/bin/bash
#SBATCH --job-name=TERRA_smoother_train
#SBATCH --partition=a100-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time 24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sturm113@umn.edu

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load miniforge

# Activate your virtual environment (adjust path as needed).
source activate /projects/standard/runke001/shared/diffusion/.conda/envs/bedrock

# Make sure the logs and models directories exist.
mkdir -p logs

# ── Run ───────────────────────────────────────────────────────────────────────
python -u train_smoother.py

echo "Job finished at $(date)"