# Automatically run the both training and validation scripts 
# for the transformer and LSTM models

# Usage: python main.py --model-choice transformer --train-folder <folder that contains "config.py">

import argparse
import os
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-choice', type=str, required=True, choices=['transformer', 'lstm'])
    parser.add_argument('--train-folder', type=str, required=True)

    args = parser.parse_args()

    if args.model_choice == 'transformer':
        # Run the training script
        command = f'./example-train.sh {args.train_folder}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True)

        # Run the validation script
        os.environ['MODEL_CHOICE'] = 'transformer'
        command = f'python validation.py --csv-folder data/yelp-polarity --output-dir {args.train_folder}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True, env=os.environ.copy())

    elif args.model_choice == 'lstm':
        # Run the training script
        command = f'./example-train-lstm.sh {args.train_folder}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True)

        # Run the validation script
        os.environ['MODEL_CHOICE'] = 'lstm'
        command = f'python validation.py --csv-folder data/yelp-polarity --output-dir {args.train_folder}'
        print(f'Running command: {command}')
        subprocess.run(command, shell=True, env=os.environ.copy())
