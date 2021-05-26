#!/bin/bash
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv


python3 ../analyze/analyze_retinanet_kitti.py --augment horizontal_shift --backbone vgg19


conda deactivate
