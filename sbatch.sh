#!/bin/bash
#SBATCH --job-name=TimeMoE_Mamba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-04:00:00
#SBATCH --mail-user=aaburkova_1@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
set -x

# Use env python directly (avoid conda activate issues under modules)
ENV=/home/aaburkova_1/.conda/envs/mama
PY=$ENV/bin/python

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# Quick sanity: show torch + cuda visibility
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("current:", torch.cuda.current_device())
    print("name:", torch.cuda.get_device_name(0))
PY

# Mamba CUDA deps must be installed on the LOGIN node (compute nodes often have no internet).
# Run once before submitting jobs that use --temporal_mixer mamba:
#   conda activate mama
#   pip install causal-conv1d einops
#   pip install -e /home/aaburkova_1/project-1/black_mamba

# Scratch dir fallback (some clusters don't set SLURM_TMPDIR)
SCRATCH_DIR=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$SCRATCH_DIR"

# Data: copy to node-local storage for speed
DATA_SRC=/home/aaburkova_1/project-1/dataset_finance_fred_features/finetune/train.jsonl
DATA_DST=$SCRATCH_DIR/train_sequences.jsonl
ls -lh "$DATA_SRC"
cp "$DATA_SRC" "$DATA_DST"
ls -lh "$DATA_DST"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Comet ML (trainer will pick these up)
export COMET_API_KEY="0Oz9BVkChPlRpa0Zh4OOQubGp"
export COMET_PROJECT_NAME="time-moe-tuning"
export COMET_WORKSPACE="s22d-burkova"
export COMET_EXPERIMENT_NAME="time_moe_mamba_fred_features"

# IMPORTANT: run through the repo wrapper for single-node multi-GPU
# (It will auto-call torchrun with nproc_per_node = visible GPUs)
srun "$PY" /home/aaburkova_1/project-1/torch_dist_run.py \
  /home/aaburkova_1/project-1/main.py \
  -d "$DATA_DST" \
  --from_scratch \
  --train_steps 50000 \
  --output_path /home/aaburkova_1/project-1/logs/time_moe_mamba_fred_features \
  --dataloader_num_workers 2 \
  --precision bf16 \
  --gradient_checkpointing \
  --micro_batch_size 8 \
  --global_batch_size 32 \
  --evaluation_strategy no \
  --temporal_mixer mamba \
  --use_covariates \
  --main_input_size 11 \
  --macro_input_size 6 \
  --macro_fusion add
