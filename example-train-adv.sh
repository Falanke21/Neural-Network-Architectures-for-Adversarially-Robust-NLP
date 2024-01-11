#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
#SBATCH -p gpgpuB
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

# Important Note: All adversarial training is 1 epoch! for now
# Important Note: we use dataset data/split-for-adv-train! for now

# Usage: example-train-adv.sh <output folder> <victim_model>
# Example: example-train-adv.sh adv/baseline adv/baseline/victim_model.pt 

# For every data/split-for-adv-train/{i} folder, we iteratively train the model
# and use the output model of last iteration as the input of next iteration.

# Note: we do split-for-adv-train because of the memory limit of machines
# The splitting is done by utils/split_csv.py
# Note: the victim model needs to be copied to the output folder

# check if all arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <output folder> <victim_model>"
    exit 1
fi

export MODEL_CHOICE="transformer"

# First copy the victim model to be named "at_model.pt"
cp $2 $1/at_model.pt

# make dir checkpoints if not exist
mkdir -p $1/checkpoints

# Note: multiple epochs adv training implementation is here, not in adversarial.py
# get the NUM_ADV_EPOCHS value from <output folder>/config.py, and store it in a variable
cd $1
NUM_ADV_EPOCHS="$(python - <<END
import config
try:
    print(config.TransformerConfig.NUM_ADV_EPOCHS)
except AttributeError:
    print(1)
END
)"
cd -

echo "NUM_ADV_EPOCHS: $NUM_ADV_EPOCHS"

# train all data
for j in $(seq 1 $NUM_ADV_EPOCHS)
do
  echo
  echo "#### Training epoch $j/$NUM_ADV_EPOCHS"
  for i in {1..10}
  do
      echo
      echo "#### Training iteration $i/10, on data/split-for-adv-train/$i"
      echo
      python train.py --csv data/split-for-adv-train/$i \
        --loss-values \
        --adversarial-training \
        --output-dir $1 \
        --load-trained $1/at_model.pt

      # "at_model.pt" is now a adversarially trained model, we will save it to checkpoints
      # the number of the model is <(j-1)*10+(i-1)>
      k=$((j-1))
      q=$((i-1))
      n=$(($k*10+$q))
      # n = n + 1, we want to save the model as at_model_1.pt, at_model_2.pt, ...
      n=$((n+1))
      cp $1/at_model.pt $1/checkpoints/at_model_$n.pt
  done
done

# finally remove 'at_model.pt'
rm $1/at_model.pt
