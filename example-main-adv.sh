#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422

# get 2 command line arguments
# 1. model choice
# 2. train folder
# 3. adversarial start model

# Usage: example-main-adv.sh <model choice> <train folder> <adversarial start model>

python main.py --model-choice $1 --train-folder $2 --adversarial --adversarial-start-model $3
