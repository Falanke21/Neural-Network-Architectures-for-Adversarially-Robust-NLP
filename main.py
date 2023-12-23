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
    parser.add_argument('--adversarial', action='store_true', default=False, 
                        help='Whether this is an adversarial training run')
    parser.add_argument('--adversarial-start-model', type=str, default=None,
                        help='Path to the model to start adversarial training from')

    args = parser.parse_args()

    if args.adversarial:
        # Adversarial training
        if args.model_choice == 'transformer':
            command = f'./example-train-adv.sh {args.train_folder} {args.adversarial_start_model}'
            print(f'Running command: {command}')
            subprocess.run(command, shell=True)

            # Run the validation script
            os.environ['MODEL_CHOICE'] = 'transformer'
            # set --every-n to 1 to validate on all models in the checkpoints folder
            command = f'python validation.py --csv-folder data/yelp-polarity --output-dir {args.train_folder} --adversarial'
            print(f'Running command: {command}')
            subprocess.run(command, shell=True, env=os.environ.copy())
        elif args.model_choice == 'lstm':
            raise NotImplementedError('Adversarial training for LSTM not implemented')
    else: 
        # Normal training, not adversarial
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
