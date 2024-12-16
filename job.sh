#!/bin/bash
#SBATCH --job-name=vbc_ffs
#SBATCH --partition=a5000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1 
#SBATCH --mem=32G
#SBATCH --time=24:00:00  
#SBATCH --gres=gpu:nvidia_rtx_a5000:2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate py39
python vbcFFS.py