#!/bin/bash

#SBATCH --job-name=lstmr
#SBATCH -o /data2/npl/ViInfographicCaps/workspace/baseline/LSTMR/lstmr.out
#SBATCH --error=lstmr_error.out
#SBATCH --gres=gpu:1
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
python main.py \
--config ./config/lstmr_config_yolo.yaml \
--save_dir ./save \
--run_type train \
--device cuda:0 \
--resume_file /data2/npl/ViInfographicCaps/workspace/baseline/LSTMR/save/checkpoints/model_last.pth
