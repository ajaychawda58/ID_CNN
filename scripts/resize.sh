#!/bin/bash
#SBATCH -t 240
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH -N 1

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate idenv
python3 ../resize.py

conda deactivate
