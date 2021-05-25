#!/bin/bash
#SBATCH -t 180               # time limit set to 15 minutes
#SBATCH --mem=16G          # 4G of memory are reserved
#SBATCH -n 1                 # 1 processor to be used
#SBATCH -N 1                 # 1 node is used 

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv

python3 ../augmentation.py

conda deactivate
