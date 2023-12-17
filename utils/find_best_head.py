# Given a directory path, find the best head configuration in the subdirectories
# By examine the model_selection_result.txt files in each subdirectory

# Example input path: vol_folder/model_zoo/continue/4-layer/trial1/tran/nreva
# Example txt file: vol_folder/model_zoo/continue/4-layer/trial1/tran/nreva/3head/model_selection_result.txt
# Example content in txt file: Standard accuracy and accuracy under attack for 1.: (0.9327631578947368, 0.46799999999999997)
# We want the two numbers in the tuple, (best_std_acc, best_acc_under_attack)

import argparse
import os

HEAD_CONFIGS = ["3head", "5head", "10head", "15head", "20head", "30head"]


def find_accs_from_model_result_txt(result_file: str) -> tuple:
    """
    # Given an example txt path, get the accuracies in the subdirectories
    # By examine the model_selection_result.txt files in each subdirectory

    # Example txt file: vol_folder/model_zoo/continue/4-layer/trial3/tran/nreva/20head/model_selection_result.txt
    # Example content in txt file: Standard accuracy and accuracy under attack for 1.: (0.9327631578947368, 0.46799999999999997)
    # We want the two numbers in the tuple, (best_std_acc, best_acc_under_attack)
    """
    with open(result_file, "r") as f:
        # Find the second line
        f.readline()
        target_line = f.readline()
        # Find the tuple
        tuple_start = target_line.find("(")
        tuple_end = target_line.find(")")
        tuple_str = target_line[tuple_start+1:tuple_end]
        # Find the two numbers
        numbers = tuple_str.split(",")
        std_acc = float(numbers[0])
        acc_under_attack = float(numbers[1])
        return (std_acc, acc_under_attack)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "rootdir", help="Root directory of the model architecture")
    args = arg_parser.parse_args()

    best_head = None
    best_sum_acc = 0

    for head in HEAD_CONFIGS:
        head_dir = os.path.join(args.rootdir, head)
        print(f"Checking {head_dir}")
        result_file = os.path.join(head_dir, "model_selection_result.txt")
        if not os.path.isfile(result_file):
            raise ValueError(f"File {result_file} does not exist")

        std_acc, acc_under_attack = find_accs_from_model_result_txt(
            result_file)
        sum_acc = std_acc + acc_under_attack
        print(
            f"Heads: {head}, std_acc: {std_acc}, acc_under_attack: {acc_under_attack}, sum_acc: {sum_acc}")

        if best_head is None:
            best_head = head
            best_sum_acc = std_acc + acc_under_attack
        else:
            if sum_acc > best_sum_acc:
                best_head = head
                best_sum_acc = sum_acc

    print(f"\nArch: {args.rootdir}")
    print(f"\nBest head: {best_head}, best sum acc: {best_sum_acc}")
