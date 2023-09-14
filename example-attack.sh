#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fh422
source /vol/bitbucket/fh422/venv/bin/activate
cd /homes/fh422/ic/project

if [ -z "$1" ]; then
  echo "Error: Missing argument. Please provide the path to the victim model."
  echo "Usage: $0 <path_to_victim_model>"
  exit 1
fi
# Check if the provided file exists
if [ ! -f "$1" ]; then
  echo "Error: The provided victim model file does not exist."
  exit 1
fi

# the model to attack, for ta_model_loader.py
export MODEL_CHOICE="transformer"
export TA_VICTIM_MODEL_PATH=$1
TA_VICTIM_MODEL_DIR=$(dirname "$TA_VICTIM_MODEL_PATH")

export TA_VICTIM_MODEL_EPOCH=$(basename $TA_VICTIM_MODEL_PATH | cut -d '_' -f 3)  # epoch50.pt
echo "Epoch: $TA_VICTIM_MODEL_EPOCH"
export TA_VICTIM_MODEL_EPOCH=$(basename "$TA_VICTIM_MODEL_EPOCH" ".pt")  # epoch50
echo "Attacking model $TA_VICTIM_MODEL_PATH"

# textattack
NUM_EXAMPLES=1000
# for ATTACK in textbugger ;
for ATTACK in textbugger textfooler bae deepwordbug pwws a2t ;
do
    echo ""
    echo "Running attack $ATTACK"
    export TA_ATTACK_RECIPE=$ATTACK
    echo ""
    textattack attack \
    --model-from-file ta_model_loader.py \
    --dataset-from-file ta_data_loader.py \
    --model-batch-size 32 \
    --num-examples ${NUM_EXAMPLES} \
    --disable-stdout \
    --attack-recipe $ATTACK \
    --log-individual-attack-results ${TA_VICTIM_MODEL_DIR}/attack_details/${TA_VICTIM_MODEL_EPOCH} \
    | python utils/ta_output_parser.py
done
