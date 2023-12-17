# This file is used for calculating the overall robustness accuracy of a 
# given model. The model is considered overall robust if it is robust to 
# all of the 6 attacks. Thus the overall robustness is calculated as the 
# percentage of model output samples that are robust to all 6 attacks.

import argparse
import csv
import os

ATTACKS = ['textbugger', 'textfooler', 'bae', 'deepwordbug', 'pwws', 'a2t']


def calculate_overall_robustness(output_dir: str, model_name: str):
    epoch_name = model_name.split('_')[-1].split('.')[0]  # e.g. 'epoch44'
    # first get the number of samples by counting the number of rows in
    # one of the csv files
    num_samples = 0
    with open(f'{output_dir}/attack_details/{epoch_name}/textbugger.csv', 'r') as f:
        reader = csv.reader(f)
        for _ in reader:
            num_samples += 1
    print(f'Number of testing samples: {num_samples}')

    # initialize the accuracy across attacks to 1 (all samples are robust)
    accuracy_across_attacks = [1] * num_samples

    # read in the results from the 6 attacks
    for i, attack in enumerate(ATTACKS):
        csv_path = f'{output_dir}/attack_details/{epoch_name}/{attack}.csv'
        # read in the results from each attack
        # each row is one of ['s', 'f', 'k'] where s indicates success,
        # f indicates failure, and k indicates skipped
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                # the sample is robust only if all 6 attacks are failures
                if row[0] == 's' or row[0] == 'k':
                    accuracy_across_attacks[j] = 0
        print(f"{i+1}/{len(ATTACKS)}: After {attack}, overall robustness accuracy: "
              f"{sum(accuracy_across_attacks) / num_samples * 100:.2f}%")

    # calculate the overall robustness accuracy
    overall_robustness = sum(accuracy_across_attacks) / num_samples * 100
    print(f'Overall robustness accuracy: {overall_robustness:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()
    output_dir = os.path.dirname(args.model_path)
    model_name = args.model_path.split('/')[-1]

    calculate_overall_robustness(output_dir, model_name)
