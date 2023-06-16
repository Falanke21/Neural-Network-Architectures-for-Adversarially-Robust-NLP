import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def json_to_csv(num_records_per_iteration=10000):
    current_records = 0
    with open(args.json, 'r') as f:
        df = pd.DataFrame()
        for line in f:
            data = json.loads(line)

            # only keep the stars field and the text field in the data dict, drop the rest
            data = {k: data[k] for k in ['stars', 'text']}
            # change the star field to be 0 or 1, where [4-5] is 1, and [1-2] is 0, and [3] is ignored
            if data['stars'] >= 4:
                data['stars'] = 1
            elif data['stars'] <= 2:
                data['stars'] = 0
            else:
                continue
            # change the name of the stars field to label
            data['label'] = data.pop('stars')

            # df = df.append(data, ignore_index=True)  # deprecated by pandas
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            if len(df) > num_records_per_iteration:  # process 10000 records at a time
                df.to_csv(args.csv, mode='a', index=False, header=True)
                df = pd.DataFrame()
                current_records += num_records_per_iteration
                print(
                    f"Processed {current_records} out of {args.total_records} records")
            if current_records >= args.total_records:
                print(f"Total records processed: {current_records}")
                break

        # # process any remaining records
        # if not df.empty:
        #     df.to_csv(args.csv, mode='a', index=False, header=True)


def train_val_test_split():
    print(f"Splitting {args.csv} into train, val, test")
    df = pd.read_csv(args.csv)
    # there are some rows with label = 'label', we need to remove them
    df = df[df['label'] != 'label']
    # convert label to int
    df['label'] = df['label'].astype(int)

    # split data into train, val, test (80%, 10%, 10%)
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(
        test_data, test_size=0.5, random_state=42)

    # save the data to files
    train_data.to_csv(f'{args.csv}_train.csv', index=False, header=True)
    val_data.to_csv(f'{args.csv}_val.csv', index=False, header=True)
    test_data.to_csv(f'{args.csv}_test.csv', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--total-records', type=int, default=200000)
    args = parser.parse_args()

    json_to_csv()
    train_val_test_split()
