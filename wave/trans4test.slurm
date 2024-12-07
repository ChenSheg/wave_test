#!/bin/bash
#SBATCH --job-name=chenbojin_job            
#SBATCH --partition=a10
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=10G
#SBATCH --gres=gpu:1  # 分配 1 个 GPU
#SBATCH --time=47:59:59
#SBATCH --output=/slurm/home/yrd/liaolab/chenbojin/chen_v6/output_%A_%a.txt  # 输出文件
#SBATCH --error=/slurm/home/yrd/liaolab/chenbojin/chen_v6/error_%A_%a.txt    # 错误文件
#SBATCH --mail-user=790031458@qq.com
#SBATCH --mail-type=all

source /slurm/home/yrd/liaolab/chenbojin/anaconda3/etc/profile.d/conda.sh
conda activate trans4test
export PYTHONWARNINGS="ignore"

# 直接传递任务 ID
python train.py --outdir /slurm/home/yrd/liaolab/chenbojin/chen_v6/output --train_dataset /slurm/home/yrd/liaolab/chenbojin/chen_v6/split_data/Fold1/train.h5ad --val_dataset /slurm/home/yrd/liaolab/chenbojin/chen_v6/split_data/Fold1/val.h5ad --test_dataset /slurm/home/yrd/liaolab/chenbojin/chen_v6/split_data/Fold1/test.h5ad --genevae_model_path /slurm/home/yrd/liaolab/chenbojin/chen_v6/vae_model.pth
