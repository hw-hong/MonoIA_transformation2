#!/usr/bin/bash

#SBATCH -J monoia_consistency
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -t 2-0
#SBATCH -o logs/slurm-%A_monoia_consistency_10.out

bash
conda activate monoia
cd /nas2/data/heewon.hong/MonoIA_consistency
bash train.sh 0 --config config/monoia_10.yaml
exit 0
