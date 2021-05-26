#!/bin/bash
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv
python3 ../analyze/analyze_coco_vgg16.py --augment test2017
python3 ../analyze/analyze_coco_vgg16.py --augment channel_shift
python3 ../analyze/analyze_coco_vgg16.py --augment horizontal_flip
python3 ../analyze/analyze_coco_vgg16.py --augment horizontal_shift
python3 ../analyze/analyze_coco_vgg16.py --augment vertical_shift
python3 ../analyze/analyze_coco_vgg16.py --augment vertical_flip
python3 ../analyze/analyze_coco_vgg16.py --augment rotation

conda deactivate
