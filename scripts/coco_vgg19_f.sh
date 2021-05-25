#!/bin/bash
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:4

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv

python ../train.py --dataset coco --backbone vgg19 --model faster_rcnn --epoch 10

conda deactivate
