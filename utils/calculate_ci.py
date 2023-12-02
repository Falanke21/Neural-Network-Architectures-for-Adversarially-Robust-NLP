# Calculate the 95% confidence interval for our standard accuracies
# and accuracies under attacks, over 3 different runs.

import csv
import numpy as np

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
        print(f"{model}: {model_to_mean_std_acc[model]:.2f}% ± {model_to_ci_std_acc[model]:.2f}%")
    print("\nAccuracies under attack:")
    for model in model_to_mean_acc_under_attack:
        print(f"{model}: {model_to_mean_acc_under_attack[model]:.2f}% ± {model_to_ci_acc_under_attack[model]:.2f}%")

    # write into csv in .2f format
    with open('mean_n_95ci.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lstm', 'transformer', 'cosformer', 'reva', 'revcos'])
        # write standard accuracies mean
        writer.writerow([f"{model_to_mean_std_acc[model]:.2f}%" for model in model_to_mean_std_acc])
        # write accuracies under attack mean
        writer.writerow([f"{model_to_mean_acc_under_attack[model]:.2f}%" for model in model_to_mean_acc_under_attack])
        # write standard accuracies 95% confidence interval
        writer.writerow([f"{model_to_ci_std_acc[model]:.2f}%" for model in model_to_ci_std_acc])
        # write accuracies under attack 95% confidence interval
        writer.writerow([f"{model_to_ci_acc_under_attack[model]:.2f}%" for model in model_to_ci_acc_under_attack])