#!/bin/bash
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv

python ../train.py --dataset kitti --backbone vgg19 --model faster_rcnn --epoch 60

conda deactivate
