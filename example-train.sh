#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
#SBATCH -p gpgpuB
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

if [ -z "$1" ]; then
  echo "Error: Missing argument. Please provide the path to the output directory."
  echo "Usage: $0 output_dir [load_trained_model_name]"
  exit 1
fi
# Check if the provided file exists
if [ ! -d "$1" ]; then
  echo "Error: The provided output dir does not exist."
  exit 1
fi

# Check if second optional argument is provided
if [ -z "$2" ]; then
  echo "No trained model provided. Training from scratch."
else
  echo "Loading trained model $2"
  LOAD_TRAINED_MODEL=$2
fi

export MODEL_CHOICE="transformer"

# if LOAD_TRAINED_MODEL exists, then train with --load-trained
if [ -z "$LOAD_TRAINED_MODEL" ]; then
  # train the model
  python train.py --csv data/data300k-with-3stars \
      --checkpoints \
      --loss-values \
      --output-dir $1
else
  python train.py --csv data/data300k-with-3stars \
      --checkpoints \
      --loss-values \
      --output-dir $1 \
      --load-trained $LOAD_TRAINED_MODEL
fi
