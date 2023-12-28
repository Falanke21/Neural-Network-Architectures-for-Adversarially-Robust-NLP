# Calculate the 95% confidence interval for our standard accuracies
# and accuracies under attacks, over 3 different runs.

import argparse
import csv
import numpy as np
import pandas as pd
import os

from find_best_head import find_accs_from_model_result_txt

# Hard-coded values from Excel into Python dictionaries
TRIAL1_STD_ACC = {
    'lstm': 94.08,
    'transformer': 93.16,
    'cosformer': 91.70,
    'reva': 92.66,
    'revcos': 93.88
}
TRIAL1_ACC_UNDER_ATTACK = {
    'lstm': 48.89,
    'transformer': 52.91,
    'cosformer': 48.71,
    'reva': 57.59,
    'revcos': 52.84
}
TRIAL2_STD_ACC = {
    'lstm': 94.06,
    'transformer': 92.26,
    'cosformer': 91.62,
    'reva': 92.16,
    'revcos': 92.04
}
TRIAL2_ACC_UNDER_ATTACK = {
    'lstm': 48.24,
    'transformer': 48.81,
    'cosformer': 49.38,
    'reva': 54.63,
    'revcos': 49.38
}
TRIAL3_STD_ACC = {
    'lstm': 94.58,
    'transformer': 91.98,
    'cosformer': 90.80,
    'reva': 92.84,
    'revcos': 93.08
}
TRIAL3_ACC_UNDER_ATTACK = {
    'lstm': 48.08,
    'transformer': 48.34,
    'cosformer': 48.73,
    'reva': 56.20,
    'revcos': 51.29
}


def find_ci_for_5_models():
    # from the 3 trials, we first collect the accuracies for each model
    # into a dictionary of lists
    model_to_std_acc = {
        'lstm': [],
        'transformer': [],
        'cosformer': [],
        'reva': [],
        'revcos': []
    }
    model_to_acc_under_attack = {
        'lstm': [],
        'transformer': [],
        'cosformer': [],
        'reva': [],
        'revcos': []
    }
    model_to_mean_std_acc = {}
    model_to_mean_acc_under_attack = {}
    model_to_ci_std_acc = {}
    model_to_ci_acc_under_attack = {}

    # populate the 3 trials into {model: [acc1, acc2, acc3]}
    for model in model_to_std_acc:
        model_to_std_acc[model].append(TRIAL1_STD_ACC[model])
        model_to_std_acc[model].append(TRIAL2_STD_ACC[model])
        model_to_std_acc[model].append(TRIAL3_STD_ACC[model])
        model_to_acc_under_attack[model].append(TRIAL1_ACC_UNDER_ATTACK[model])
        model_to_acc_under_attack[model].append(TRIAL2_ACC_UNDER_ATTACK[model])
        model_to_acc_under_attack[model].append(TRIAL3_ACC_UNDER_ATTACK[model])

    # calculate the mean and 95% confidence interval for each model
    # then store them into the respective dictionaries
    for model in model_to_std_acc:
        mean, ci = calculate_ci(model_to_std_acc[model])
        model_to_mean_std_acc[model] = mean
        model_to_ci_std_acc[model] = ci
        mean, ci = calculate_ci(model_to_acc_under_attack[model])
        model_to_mean_acc_under_attack[model] = mean
        model_to_ci_acc_under_attack[model] = ci

    # print the results in .2f format
    print("Standard accuracies:")
    for model in model_to_mean_std_acc:
        print(
            f"{model}: {model_to_mean_std_acc[model]:.2f}% ± {model_to_ci_std_acc[model]:.2f}%")
    print("\nAccuracies under attack:")
    for model in model_to_mean_acc_under_attack:
        print(
            f"{model}: {model_to_mean_acc_under_attack[model]:.2f}% ± {model_to_ci_acc_under_attack[model]:.2f}%")

    # write into csv in .2f format
    with open('mean_n_95ci.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lstm', 'transformer', 'cosformer', 'reva', 'revcos'])
        # write standard accuracies mean
        writer.writerow(
            [f"{model_to_mean_std_acc[model]:.2f}%" for model in model_to_mean_std_acc])
        # write accuracies under attack mean
        writer.writerow(
            [f"{model_to_mean_acc_under_attack[model]:.2f}%" for model in model_to_mean_acc_under_attack])
        # write standard accuracies 95% confidence interval
        writer.writerow(
            [f"{model_to_ci_std_acc[model]:.2f}%" for model in model_to_ci_std_acc])
        # write accuracies under attack 95% confidence interval
        writer.writerow(
            [f"{model_to_ci_acc_under_attack[model]:.2f}%" for model in model_to_ci_acc_under_attack])


def find_accs_from_ta_results_csv(csv_path: str) -> tuple:
    """
    # Given an example txt path, get the accuracies
    # By examine the ta_results_***.csv file
    # Return a tuple of (best_std_acc, best_acc_under_attack)
    # Note: these returned values are in percentage form, but we remove the % sign
    """
    # Read the csv file into a dataframe
    df = pd.read_csv(csv_path)
    # make sure the dataframe is exactly 6 rows plus the header
    if len(df) != 6:
        raise ValueError(
            f"Number of rows in {csv_path} is not 6, did the test finish?")
    # first get std_acc, note it's the same for all rows
    std_acc = df.iloc[0]['Original accuracy']
    # convert from a string of percentage to a float
    std_acc = float(std_acc[:-1])
    # then get acc_under_attack, also need to convert to float first,
    # note we need to take the mean of all rows
    lst = []
    for i in range(len(df)):
        acc_under_attack = df.iloc[i]['Accuracy under attack']
        acc_under_attack = float(acc_under_attack[:-1])
        lst.append(acc_under_attack)
    acc_under_attack = np.mean(lst)
    return std_acc, acc_under_attack


def find_ci_from_rootdir(arch: str, best_head: str):
    """
    # Given an example txt path, get the accuracies and calculate the ci in the subdirectories
    # By examine the model_selection_result.txt files in each subdirectory

    # Example txt file: vol_folder/model_zoo/continue/4-layer/trial3/tran/nreva/20head/model_selection_result.txt
    # Example content in txt file: Standard accuracy and accuracy under attack for 1.: (0.9327631578947368, 0.46799999999999997)
    # We want the two numbers in the tuple, (best_std_acc, best_acc_under_attack)
    # We want to find the ci for these 2 numbers by trial 1, 2 and 3.
    """
    trial_names = ["trial1", "trial2", "trial3"]

    std_acc_lst = []
    acc_under_attack_lst = []

    #     txt_path = f"vol_folder/model_zoo/continue/4-layer/{trial_name}/tran/{arch}/{best_head}/model_selection_result.txt"
    #     std_acc, acc_under_attack = find_accs_from_model_result_txt(txt_path)
    #     # these are in decimal form, so we multiply by 100 to get percentage
    #     std_acc *= 100
    #     acc_under_attack *= 100
    #     std_acc_lst.append(std_acc)
    #     acc_under_attack_lst.append(acc_under_attack)
    for trial_name in trial_names:
        dir_path = f"vol_folder/model_zoo/continue/4-layer/{trial_name}/tran/{arch}/{best_head}"
        # search for a file name in the dir starting with "ta_results_***.csv"
        csv_path = ""
        for file_name in os.listdir(dir_path):
            if file_name.startswith("ta_results_"):
                csv_path = os.path.join(dir_path, file_name)
                break
        if csv_path == "":
            raise ValueError(f"No ta_results_***.csv file found in {dir_path}, did you run the test?")
        # These are in percentage form, but don't have the % sign
        std_acc, acc_under_attack = find_accs_from_ta_results_csv(csv_path)
        std_acc_lst.append(std_acc)
        acc_under_attack_lst.append(acc_under_attack)

    std_acc_mean, std_acc_ci = calculate_ci(std_acc_lst)
    acc_under_attack_mean, acc_under_attack_ci = calculate_ci(
        acc_under_attack_lst)

    print(f"Architecture: {arch}, best_head: {best_head}")
    print(f"Standard accuracy: {std_acc_mean:.2f}% ± {std_acc_ci:.2f}%")
    print(
        f"Accuracy under attack: {acc_under_attack_mean:.2f}% ± {acc_under_attack_ci:.2f}%")

    return std_acc_mean, std_acc_ci, acc_under_attack_mean, acc_under_attack_ci


def calculate_ci(accuracies: list) -> tuple:
    """
    # Calculate the 95% confidence interval for the given accuracies.
    """
    mean = np.mean(accuracies)
    std = np.std(accuracies)

    # Calculate the 95% confidence interval
    standard_error = std / np.sqrt(len(accuracies))
    ci = 1.96 * standard_error

    return mean, ci


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", help="Architecture of the model")
    parser.add_argument("best_head", help="Best head of the model")
    args = parser.parse_args()

    find_ci_from_rootdir(args.arch, args.best_head)
    # find_ci_for_5_models()
