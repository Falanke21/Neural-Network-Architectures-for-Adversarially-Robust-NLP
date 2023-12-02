#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
#SBATCH -p gpgpuB
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

# Usage: clusterjob-train-adv.sh <output folder> <victim_model>
# Example: clusterjob-train-adv.sh adv/baseline adv/baseline/victim_model.pt data/split-for-adv-train

# For every data/split-for-adv-train/{i} folder, we iteratively train the model
# and use the output model of last iteration as the input of next iteration.

export MODEL_CHOICE="transformer"

# First copy the victim model to be named "at_model.pt"
cp $2 $1/at_model.pt

# make dir checkpoints
mkdir -p $1/checkpoints
# # We only train for 5 iterations for now
# for i in {0..4}
# train all data
for i in {0..9}
do
    echo
    echo "#### Training iteration $i, on data/split-for-adv-train/$i"
    echo
    python train.py --csv data/split-for-adv-train/$i \
      --loss-values \
      --adversarial-training \
      --output-dir $1 \
      --load-trained $1/at_model.pt

    # "at_model.pt" is now a adversarially trained model, we will save it to checkpoints
    cp $1/at_model.pt $1/checkpoints/at_model_$i.pt
done

# finally cp 'at_model.pt' to be 'transfromer'
cp $1/at_model.pt $1/transformer_model_at.pt