import argparse
import csv
import pandas as pd
import os
import subprocess
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.yelp_review_dataset import YelpReviewDataset
from utils.model_factory import construct_model_from_config
from utils.ta_output_parser import parse_ta_output, get_acc_under_attack


def get_standard_val_acc(epoch, val_dataset, Config, model, device):
    # try to find {model_choice}_val_accuracy.txt from output_dir
    # if found, we can skip the validation process
    # and search for the line number of the current epoch
    # if not found, we need to do the validation process
    if os.path.exists(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt'):
        print(f"Using epoch {epoch} accuracy from \
            {args.output_dir}/{os.environ['MODEL_CHOICE']}_val_accuracy.txt")
        with open(f'{args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # i starts from 0, but epoch starts from 1
                if i + 1 == epoch:
                    return float(line)
        # we should never reach here
        raise ValueError(
            f'Could not find epoch {epoch} in \
            {args.output_dir}/{os.environ["MODEL_CHOICE"]}_val_accuracy.txt')

    else:
        print(f"Could not find {args.output_dir}/{os.environ['MODEL_CHOICE']}_val_accuracy.txt")
        print("Running validation process...")
        # otherwise, we need to do the validation process
        # get dataloader from dataset
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        # val
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            TP, TN = 0, 0
            for data, labels, _ in tqdm(val_loader):
                data = data.to(device)
                labels = labels.unsqueeze(1).float().to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)

                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
            print(f"Validation Accuracy: {(TP + TN) / total:.4f}")
            print(f"Validation Loss: {total_loss / len(val_loader):.4f}")

        standard_val_acc = (TP + TN) / total
        return standard_val_acc


def run_ta_calulate_acc_under_attack(model_path) -> float:
    """
    Run textattack attack command of one epoch model
    and calculate the accuracy under attack
    Return the accuracy under attack as float
    Note: only run textfooler attack
    """
    # first set TA_VICTIM_MODEL_PATH to the model path
    os.environ['TA_VICTIM_MODEL_PATH'] = model_path

    # run the specific textattack command
    attack = "textfooler"
    num_attack_examples = 1000
    query_budget = 300
    command = f"textattack attack \
        --model-from-file ta_model_loader.py \
        --dataset-from-file ta_data_loader_validation.py \
        --model-batch-size 32 --num-examples {num_attack_examples} \
        --disable-stdout --attack-recipe {attack} \
        --query-budget {query_budget}"
    print(f'Running command: {command}')
    ta_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    text_stream = ta_process.stdout.decode('utf-8').split('\n')
    # parse textattack output and get the accuracy under attack for this epoch
    data = parse_ta_output(text_stream)
    return get_acc_under_attack(data)


def calculate_all_validation_results(Config, args, checkpoint_dir, vocab, model, device):
    """
    Top down function to calculate the validation results of every 5 epochs
    """
    # create a dict to accumulate the results of every 5 epochs
    # dict key: epoch, value: (standard accuracy, accuracy under textfooler)
    validation_results = {}
    total_epochs = Config.NUM_EPOCHS
    # create a csv file with header to store the results of every 5 epochs
    with open(f'{args.output_dir}/model_selection_result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Epoch",
                "Standard validation accuracy",
                "Accuracy under attack",
            ]
        )
    for epoch in range(5, total_epochs + 1, 5):
        print(f'\n#####\nValidating epoch {epoch}/{total_epochs}\n#####\n')
        model_path = f'{checkpoint_dir}/{os.environ["MODEL_CHOICE"]}_model_epoch{epoch}.pt'
        print(f"Loading model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
        # if the epoch is not found, we simply skip it
        except FileNotFoundError:
            print(f"Could not find {model_path}, skipping...")
            continue
        model.eval()

        val_data = pd.read_csv(f'{args.csv_folder}/val.csv')
        # Reset dataframe index so that we can use df.loc[idx, 'text']
        val_data = val_data.reset_index(drop=True)
        val_dataset = YelpReviewDataset(val_data, vocab, Config.MAX_SEQ_LENGTH)

        # calculate the standard accuracy for this epoch
        float_standard_val_acc = get_standard_val_acc(
            epoch, val_dataset, Config, model, device)

        # now validate the accuracy under attack (textfooler)
        float_acc_under_attack = run_ta_calulate_acc_under_attack(model_path)

        validation_results[epoch] = (float_standard_val_acc, float_acc_under_attack)
        # write the results of each epoch to the csv file
        with open(f'{args.output_dir}/model_selection_result.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    float_standard_val_acc,
                    float_acc_under_attack,
                ]
            )
    return validation_results


def find_best_epochs(validation_results):
    """
    Find the best epoch based on 2 strategies:
    1. Based on the sum of standard accuracy and accuracy under attack
    2. Based on standard accuracy
    Return a tuple of the best epochs of the 2 strategies
    (best_epoch_of_sum, best_epoch_of_standard)
    """
    best_epoch_of_sum = 0
    max_sum_of_accuracy = 0
    for epoch, (standard_val_acc, acc_under_attack) in validation_results.items():
        sum_of_accuracy = standard_val_acc + acc_under_attack
        if sum_of_accuracy > max_sum_of_accuracy:
            max_sum_of_accuracy = sum_of_accuracy
            best_epoch_of_sum = epoch

    # also we want to find the best standard accuracy epoch
    best_epoch_of_standard = 0
    max_standard_accuracy = 0
    for epoch, (standard_val_acc, acc_under_attack) in validation_results.items():
        if standard_val_acc > max_standard_accuracy:
            max_standard_accuracy = standard_val_acc
            best_epoch_of_standard = epoch
    return (best_epoch_of_sum, best_epoch_of_standard)



if __name__ == "__main__":
    assert os.environ["MODEL_CHOICE"] in [
        "lstm", "transformer"], "env var MODEL_CHOICE must be either lstm or transformer"
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='tmp')
    args = parser.parse_args()

    # default config file to output_dir/config.py
    config_path = f'{args.output_dir}/config.py'

    checkpoint_dir = f'{args.output_dir}/checkpoints'
    print(f'Checkpoint directory: {checkpoint_dir}')

    model, Config, vocab, device = construct_model_from_config(config_path)

    # calculate the validation results of every 5 epochs
    validation_results = calculate_all_validation_results(
        Config, args, checkpoint_dir, vocab, model, device)

    # in the end, we find the best epoch based on the sum of two metrics:
    best_epoch_of_sum, best_epoch_of_standard = find_best_epochs(validation_results)
    
    # output the best epochs and their accuracy to a txt file in output_dir
    with open(f'{args.output_dir}/model_selection_result.txt', 'w') as f:
        f.write(f'1. Best epoch based on sum of standard accuracy and accuracy under attack: {best_epoch_of_sum}\n')
        f.write(f'Standard accuracy and accuracy under attack for 1.: {validation_results[best_epoch_of_sum]}\n')
        f.write(f'2. Best epoch based on standard accuracy: {best_epoch_of_standard}\n')
        f.write(f'Standard accuracy and accuracy under attack for 2.: {validation_results[best_epoch_of_standard]}\n')

    # print the best epochs and their accuracy
    with open(f'{args.output_dir}/model_selection_result.txt', 'r') as f:
        print(f.read())

    # copy the best epoch of sum from checkpoints to output_dir
    model_path = f'{checkpoint_dir}/{os.environ["MODEL_CHOICE"]}_model_epoch{best_epoch_of_sum}.pt'
    subprocess.run(f'cp {model_path} {args.output_dir}', shell=True)
