import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tokenizer import MyTokenizer
from tqdm import tqdm
import numpy as np
import os


def analyse_csv(args):
    print("Analysing csv data...")
    df = pd.read_csv(args.csv)
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
            df.loc[i, 'num_tokens'] = len(
                tokenizer.tokenize(df.loc[i, 'text']))

        # plot histogram of review lengths as number of tokens
        df['num_tokens'].hist(bins=30)
        plt.title('Histogram of Review Lengths')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Number of Reviews')
        plt.savefig('word_count_hist.png')


def analyse_embeddings(args):
    print("Analysing paragramcf embeddings...")
    word_embeddings_file = "paragram.npy"
    word_list_file = "wordlist.pickle"
    mse_dist_file = "mse_dist.p"
    cos_sim_file = "cos_sim.p"
    nn_matrix_file = "nn.npy"
    word_embeddings_folder = args.paragramcf
    # Concatenate folder names to create full path to files.
    word_embeddings_file = os.path.join(
        word_embeddings_folder, word_embeddings_file
    )
    word_list_file = os.path.join(word_embeddings_folder, word_list_file)
    mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
    cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
    nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

    embedding_matrix = np.load(word_embeddings_file)
    word2index = np.load(word_list_file, allow_pickle=True)
    index2word = {}
    for word, index in word2index.items():
        index2word[index] = word
    nn_matrix = np.load(nn_matrix_file)

    # print type of each variable
    print(f"Type of embedding_matrix: {type(embedding_matrix)}")
    print(f"Type of word2index: {type(word2index)}")
    print(f"Type of index2word: {type(index2word)}")
    print(f"Type of nn_matrix: {type(nn_matrix)}")

    # print shape of each variable
    print(f"Shape of embedding_matrix: {embedding_matrix.shape}")
    print(f"Shape of word2index: {len(word2index)}")
    print(f"Shape of index2word: {len(index2word)}")
    print(f"Shape of nn_matrix: {nn_matrix.shape}")


if __name__ == "__main__":
    # load csv data
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str)
    parser.add_argument('--plot-len-hist', action='store_true', default=False)
    parser.add_argument('--paragramcf', type=str)
    args = parser.parse_args()

    if args.csv:
        analyse_csv(args)
    if args.paragramcf:
        analyse_embeddings(args)
