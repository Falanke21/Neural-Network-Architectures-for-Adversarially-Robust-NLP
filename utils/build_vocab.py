import pandas as pd
import pickle
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from tokenizer import tokenize


def yield_tokens(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        yield tokenize(row['text'])


df = pd.read_csv('data10k.csv')
# df format is:
# stars, text
# sample_text = df.iloc[0]['text']
# sample_star = df.iloc[0]['stars']
# print(sample_text)
# print(tokenize(sample_text))

# Build Vocab
token_generator = yield_tokens(df)
print("start building vocab")
vocab = build_vocab_from_iterator(token_generator, specials=['<unk>', '<pad>'])
# save vocab
with open('vocab10k.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# # load vocab
# with open('vocab10k.pkl', 'rb') as f:
#     vocab = pickle.load(f)
# print(vocab.get_stoi())
