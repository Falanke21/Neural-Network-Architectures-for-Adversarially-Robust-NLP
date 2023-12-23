import argparse
import matplotlib.pyplot as plt
import numpy as np


def do_plot(model_choice, epoch_end, output_dir):
    # load loss and accuracy values
    train_losses = []
    val_losses = []
    val_accuracy = []

    try:
        with open(f'{output_dir}/{model_choice}_train_losses.txt', 'r') as f:
            for line in f:
                train_losses.append(float(line))
        with open(f'{output_dir}/{model_choice}_val_losses.txt', 'r') as f:
            for line in f:
                val_losses.append(float(line))

        if epoch_end < len(train_losses):
            train_losses = train_losses[:epoch_end]
            val_losses = val_losses[:epoch_end]

        # plot loss
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Val loss')
        plt.xticks(np.arange(len(train_losses)),
                   np.arange(1, len(train_losses)+1))  # epoch numbers start at 1
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # avoid overlapping x-axis labels
        plt.locator_params(axis='x', nbins=20)
        plt.legend(frameon=False)
        plt.savefig(f'{output_dir}/{model_choice}_loss.png')
        # clear plot for next plot
        plt.clf()
        print(f"Loss plot saved to {output_dir}/{model_choice}_loss.png")
    except FileNotFoundError as e:
        print(f"Failed to find loss files: {e}")

    try:
        with open(f'{output_dir}/{model_choice}_val_accuracy.txt', 'r') as f:
            for line in f:
                val_accuracy.append(float(line))
        if epoch_end < len(train_losses):
            val_accuracy = val_accuracy[:epoch_end]

        # plot accuracy
        plt.plot(val_accuracy, label='Val accuracy')
        plt.xticks(np.arange(len(val_accuracy)),
                   np.arange(1, len(val_accuracy)+1))  # epoch numbers start at 1
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # avoid overlapping x-axis labels
        plt.locator_params(axis='x', nbins=20)
        plt.legend(frameon=False)
        plt.savefig(f'{output_dir}/{model_choice}_accuracy.png')
        print(
            f"Accuracy plot saved to {output_dir}/{model_choice}_accuracy.png")
    except FileNotFoundError as e:
        print(f"Failed to find accuracy file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-choice', type=str,
                        choices=['lstm', 'transformer'], required=True)
    parser.add_argument('--epoch-end', type=int, help='Epoch to end plot at')
    parser.add_argument('--output-dir', type=str, default='../models')
    args = parser.parse_args()

    do_plot(args.model_choice, args.epoch_end, args.output_dir)
