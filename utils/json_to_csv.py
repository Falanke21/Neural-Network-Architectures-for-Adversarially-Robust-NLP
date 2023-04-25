import argparse
import json
import pandas as pd


def main(num_records_per_iteration=10000):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        df = pd.DataFrame()
        for line in f:
            data = json.loads(line)

            # only keep the stars field and the text field in the data dict, drop the rest
            data = {k: data[k] for k in ['stars', 'text']}
            # change the star field to be 0 or 1, where greater or equal to 4 is 1, and less than 4 is 0
            data['stars'] = 1 if data['stars'] >= 4 else 0

            df = df.append(data, ignore_index=True)
            if len(df) > num_records_per_iteration:  # process 10000 records at a time
                df.to_csv(args.csv, mode='a', index=False, header=True)
                df = pd.DataFrame()
        
        # process any remaining records
        if not df.empty:
            df.to_csv(args.csv, mode='a', index=False, header=False)

if __name__ == '__main__':
    main()
