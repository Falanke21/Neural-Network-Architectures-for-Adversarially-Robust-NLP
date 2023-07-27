# This file is used to parse the output of TextAttack

import csv
import fileinput
import os

# Environment variables
attack_recipe = os.environ["TA_ATTACK_RECIPE"]
model_path = os.environ["TA_VICTIM_MODEL_PATH"]
epoch_num = os.environ["TA_VICTIM_MODEL_EPOCH"]
output_dir = model_path[:model_path.rfind("/")]

original_accuracy = None
accuracy_under_attack = None
attack_success_rate = None
avg_perturbed_word = None

# parse the output of TextAttack
for line in fileinput.input():
    print(line, end="")
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

csv_filename = f"{output_dir}/ta_results_{epoch_num}.csv"
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
