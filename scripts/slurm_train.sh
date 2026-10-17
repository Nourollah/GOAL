#!/bin/bash
#SBATCH --job-name=goal-train
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
# GOAL SLURM training script
#
# Supports automatic resumption via last.ckpt and TRAINING_COMPLETE sentinel.
# SLURM --requeue + --open-mode=append allow transparent preemption handling.
# ---------------------------------------------------------------------------

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
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

# Experiment = one self-contained file in src/goal/configs/ml/ (override
# with EXPERIMENT=<name> sbatch scripts/slurm_train.sh).  Devices and nodes
# must match the #SBATCH lines above; strategy.name=ddp shards the batches.
EXPERIMENT="${EXPERIMENT:-simurgh_gmd26}"

srun pixi run -e cuda python -m goal.ml.cli.train \
    --config-name "$EXPERIMENT" \
    trainer.accelerator=gpu \
    trainer.num_nodes="$SLURM_NNODES" \
    trainer.devices="$SLURM_NTASKS_PER_NODE" \
    strategy.name=ddp \
    training.slurm_mode=true

echo "Finished or preempted at $(date)"
