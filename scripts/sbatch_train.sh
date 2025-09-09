#!/bin/bash

#SBATCH --job-name=lstmr
#SBATCH -o /data2/npl/ViInfographicCaps/workspace/baseline/LSTMR/lstmr.out
#SBATCH --error=lstmr_error.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1000:00:00

source /data2/npl/ViInfographicCaps/scripts/activate_global.sh

which python

echo "===== GPU Status (nvidia-smi) ====="
nvidia-smi

echo "===== Checking PyTorch CUDA availability ====="
python3 - <<EOF
import torch
print("Torch CUDA available? ", torch.cuda.is_available())
EOF

echo "===== Training ====="
cd /data2/npl/ViInfographicCaps/workspace/baseline/LSTMR

source /data2/npl/ViInfographicCaps/workspace/baseline/LSTMR/scripts/train.sh
