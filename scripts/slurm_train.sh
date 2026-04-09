#!/bin/bash
#SBATCH --job-name=gmd-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --output=logs/slurm/%J.out
#SBATCH --error=logs/slurm/%J.err
#SBATCH --requeue
#SBATCH --open-mode=append

# ---------------------------------------------------------------------------
# GMD SLURM training script
#
# Supports automatic resumption via last.ckpt and TRAINING_COMPLETE sentinel.
# SLURM --requeue + --open-mode=append allow transparent preemption handling.
# ---------------------------------------------------------------------------

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN

echo "====================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Nodes     : $SLURM_JOB_NODELIST"
echo "Start     : $(date)"
echo "====================================="

# Create log directory for SLURM output
mkdir -p logs/slurm

srun pixi run -e cuda python -m gmd.cli.train \
    model=hyperspec \
    data=xyz \
    training.num_nodes=$SLURM_NNODES \
    training.devices=4 \
    training.slurm_mode=true

echo "Finished or preempted at $(date)"
