import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tokenizer import MyTokenizer
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    # load csv data
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--plot-len-hist', action='store_true', default=False)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df['label'] != 'label']
    df['label'] = df['label'].astype(np.float64)
    # df format: stars, text
    # print first row
    print(f"First row: {df.iloc[0]}")
    print(f"First label: {df.iloc[0]['label']}")
    print(f"Type of label: {type(df.iloc[0]['label'])}")

    # print total number of rows
    print(f"Total number of rows: {len(df)}")

    # print num of positive vs negative reviews
    print(f"Number of positive reviews: {len(df[df['label'] == 1])}")
    print(f"Number of negative reviews: {len(df[df['label'] == 0])}")

    if args.plot_len_hist:
        tokenizer = MyTokenizer(None, 150, remove_stopwords=False)
        # get the length of tokenized all text, show progress bar with tqdm
        for i in tqdm(range(len(df))):
            df.loc[i, 'num_tokens'] = len(tokenizer.tokenize(df.loc[i, 'text']))

        # plot histogram of review lengths as number of tokens
        df['num_tokens'].hist(bins=30)
        plt.title('Histogram of Review Lengths')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Number of Reviews')
        plt.savefig('word_count_hist.png')
