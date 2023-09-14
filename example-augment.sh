#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
#SBATCH -p gpgpuB
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

if [ -z "$1" ]; then
  echo "Error: Missing argument. Please provide the path to the output directory."
  echo "Usage: $0 [load_trained_model_name]"
  exit 1
fi

export MODEL_CHOICE="transformer"

python utils/augment.py --csv-folder data/data300k-with-3stars \
  --load-trained $1 \
  --data-proportion 0.5 \
  --concat-with-original
