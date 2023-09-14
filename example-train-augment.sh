#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
#SBATCH -p gpgpuB
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

# Usage ./clusterjob-train-augment.sh output_dir load_trained_model_name augment_data_dir

if [ -z "$1" ]
  then
    echo "No output directory supplied"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No trained model name supplied"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No augment data directory supplied"
    exit 1
fi

echo "Starting training with adversarial augmented data"

export MODEL_CHOICE="transformer"

python train.py --csv $3 \
    --loss-values \
    --output-dir $1 \
    --load-trained $2 \
    --checkpoints
