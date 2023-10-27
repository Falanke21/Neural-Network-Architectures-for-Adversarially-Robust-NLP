# The yelp-polarity dataset is a bit different from the yelp-review database we used
# We do the following things in this script:
# 1. Add headers (label, text) to the csv files
# 2. Change labels to (0, 1) instead of (1, 2)
# 3. Split the training data into train.csv and val.csv with a ratio of 522000:38000
# 4. Copy test.csv to the output directory
# 5. Put train.csv, val.csv, test.csv in the an output directory

import argparse
import os
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='output_csv')
    args = parser.parse_args()

    # create output directory if necessary
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f'Will write outputs to "{args.output_dir}"')

    # load data
    train_data = pd.read_csv(f'{args.csv_folder}/train.csv')
    test_data = pd.read_csv(f'{args.csv_folder}/test.csv')

    # add headers (label, text) to the csv files
    train_data.columns = ['label', 'text']
    test_data.columns = ['label', 'text']

    # change labels types
    print("Changing labels to (0, 1) instead of (1, 2)")
    train_data['label'] = train_data['label'].map({1: 0, 2: 1})
    test_data['label'] = test_data['label'].map({1: 0, 2: 1})

    # For normal training, we split the training data into 2 parts: train and val
    # original train.csv (560000) -> val.csv (38000) + train.csv (522000)
    # randomly select 38000 rows from train.csv and save to val.csv
    # the rest of train.csv is the new train.csv
    # Note: there's no val.csv in the original dataset

    # randomly select 38000 rows from train.csv
    print('Split training data into train.csv and val.csv')
    val_data = train_data.sample(n=38000)
    # save to val.csv
    val_data.to_csv(f'{args.output_dir}/val.csv', index=False, header=True)
    # save the rest of train.csv to train.csv
    train_data = train_data.drop(val_data.index)
    train_data.to_csv(f'{args.output_dir}/train.csv', index=False, header=True)

    # output test.csv to the output folder
    test_data.to_csv(f'{args.output_dir}/test.csv', index=False, header=True)
