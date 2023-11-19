# This file is used to parse the output of TextAttack

import csv
import fileinput
import os


def parse_ta_output(text_stream) -> dict:
    """
    Parse the output of TextAttack from stdin
    and return a dict of the results
    """
    original_accuracy = None
    accuracy_under_attack = None
    attack_success_rate = None
    avg_perturbed_word = None

    # parse the output of TextAttack
    for line in text_stream:
        # print(line, end="")
        if "Original accuracy:" in line:
            # find the original accuracy number with the percent symbol from the line that looks like this:
            # | Original accuracy:            | 66.67% |
            original_accuracy = line.split("|")[2].strip()
        if "Accuracy under attack:" in line:
            accuracy_under_attack = line.split("|")[2].strip()
        if "Attack success rate:" in line:
            attack_success_rate = line.split("|")[2].strip()
        if "Average perturbed word %:" in line:
            avg_perturbed_word = line.split("|")[2].strip()

    # return a dict of the results
    data = {
        "original_accuracy": original_accuracy,
        "accuracy_under_attack": accuracy_under_attack,
        "attack_success_rate": attack_success_rate,
        "avg_perturbed_word": avg_perturbed_word,
    }
    return data


def write_to_csv(data: dict, output_dir: str, epoch_num: int, attack_recipe: str, ta_results_file_prefix: str = "ta_results"):
    """
    Write the results of TextAttack to a csv file
    """
    original_accuracy = data["original_accuracy"]
    accuracy_under_attack = data["accuracy_under_attack"]
    attack_success_rate = data["attack_success_rate"]
    avg_perturbed_word = data["avg_perturbed_word"]

    csv_filename = f"{output_dir}/{ta_results_file_prefix}_{epoch_num}.csv"
    # if csv_filename is not found, we initialize it with the header
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Attack Recipe",
                    "Accuracy under attack",
                    "Attack success rate",
                    "Average perturbed word %",
                    "Original accuracy",
                ]
            )

    # write the results to the csv file
    with open(csv_filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                attack_recipe,
                accuracy_under_attack,
                attack_success_rate,
                avg_perturbed_word,
                original_accuracy,
            ]
        )


def get_acc_under_attack(data):
    accuracy_under_attack = data["accuracy_under_attack"]
    # trim the percent symbol off
    accuracy_under_attack = accuracy_under_attack[:-1]
    return float(accuracy_under_attack) / 100


if __name__ == "__main__":
    # Environment variables
    attack_recipe = os.environ["TA_ATTACK_RECIPE"]
    model_path = os.environ["TA_VICTIM_MODEL_PATH"]
    epoch_num = os.environ["TA_VICTIM_MODEL_EPOCH"]
    # if exist TA_RESULTS_FILE_PREFIX, use it; otherwise, use the default value
    ta_results_file_prefix = os.environ.get("TA_RESULTS_FILE_PREFIX", "ta_results")
    output_dir = model_path[:model_path.rfind("/")]

    # parse the output of TextAttack
    data = parse_ta_output(fileinput.input())
    # write the results to a csv file
    write_to_csv(data, output_dir, epoch_num, attack_recipe, ta_results_file_prefix)
