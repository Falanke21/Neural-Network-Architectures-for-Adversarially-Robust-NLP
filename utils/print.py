import pandas as pd
import pickle

# df = pd.read_csv('data.csv')
# # print first row's text
# print(df.iloc[0]['text'])

# load vocab
with open('vocab10k.pkl', 'rb') as f:
    vocab = pickle.load(f)

print(len(vocab))
try:
    print(vocab.get_stoi()['<pad>'])
    # print(vocab.get_stoi()['<unk>'])
    # print(vocab.get_stoi()['the'])
except KeyError:
    print("KeyError")
