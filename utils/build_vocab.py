import argparse
import pandas as pd
import pickle
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from tokenizer import tokenize


def yield_tokens(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        yield tokenize(row['text'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    # df format is:
    # stars, text
    # sample_text = df.iloc[0]['text']
    # sample_star = df.iloc[0]['stars']
    # print(sample_text)
    # print(tokenize(sample_text))

    # Build Vocab
    token_generator = yield_tokens(df)
    print("start building vocab")
    vocab = build_vocab_from_iterator(token_generator, specials=['<pad>', '<unk>'], min_freq=10, max_tokens=8000)
    # save vocab
    with open(args.vocab, 'wb') as f:
        pickle.dump(vocab, f)

    # # load vocab
    # with open('vocab10k.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # print(vocab.get_stoi())
