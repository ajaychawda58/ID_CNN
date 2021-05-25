#!/bin/bash
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv


python3 ../analyze_vgg19.py --augment vertical_shift --model pascal_voc
python3 ../analyze_vgg19.py --augment vertical_flip --model pascal_voc
python3 ../analyze_vgg19.py --augment horizontal_flip --model pascal_voc
python3 ../analyze_vgg19.py --augment channel_shift --model pascal_voc
python3 ../analyze_vgg19.py --augment rotation --model pascal_voc

conda deactivate
