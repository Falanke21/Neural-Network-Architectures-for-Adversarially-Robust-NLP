#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422

# get 2 command line arguments
# 1. model choice
# 2. train folder

python main.py --model-choice $1 --train-folder $2
