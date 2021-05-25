#!/bin/bash
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate idenv

python3 ../evaluate.py --dataset coco --model faster_rcnn --backbone vgg16

conda deactivate
