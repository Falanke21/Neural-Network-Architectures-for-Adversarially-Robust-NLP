import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-choice', type=str,
                    choices=['lstm', 'transformer'], required=True)
parser.add_argument('--epoch-end', type=int, help='Epoch to end plot at')
args = parser.parse_args()

EMBEDDING = 'custom'
LAMBDA = '1e-6'
TITLE_FORMAT = f'{args.model_choice}, {EMBEDDING} embedding, weight decay={LAMBDA}'

# load loss and accuracy values
train_losses = []
val_losses = []
val_accuracy = []

with open(f'../{args.model_choice}_train_losses.txt', 'r') as f:
    for line in f:
        train_losses.append(float(line))
with open(f'../{args.model_choice}_val_losses.txt', 'r') as f:
    for line in f:
        val_losses.append(float(line))
with open(f'../{args.model_choice}_val_accuracy.txt', 'r') as f:
    for line in f:
        val_accuracy.append(float(line))

if args.epoch_end < len(train_losses):
    train_losses = train_losses[:args.epoch_end]
    val_losses = val_losses[:args.epoch_end]
    val_accuracy = val_accuracy[:args.epoch_end]

# plot loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Val loss')
plt.xticks(np.arange(len(train_losses)),
           np.arange(1, len(train_losses)+1))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{TITLE_FORMAT} Loss')
plt.legend(frameon=False)
plt.savefig(f'{args.model_choice}_loss.png')
# clear plot for next plot
plt.clf()
print(f"Loss plot saved to {args.model_choice}_loss.png")

# plot accuracy
plt.plot(val_accuracy, label='Val accuracy')
plt.xticks(np.arange(len(train_losses)),
           np.arange(1, len(train_losses)+1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{TITLE_FORMAT} Accuracy')
plt.legend(frameon=False)
plt.savefig(f'{args.model_choice}_accuracy.png')
print(f"Accuracy plot saved to {args.model_choice}_accuracy.png")
