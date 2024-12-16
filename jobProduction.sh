#!/bin/bash
#SBATCH --job-name=vbc_production
#SBATCH --partition=v100_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00  
#SBATCH --gres=gpu:v100:1
#SBATCH --account=free
#SBATCH --qos=normal

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate py39
python vbcProduction.py