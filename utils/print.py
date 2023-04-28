import pandas as pd
import pickle
from tokenizer import tokenize

df = pd.read_csv('../data/data10k.csv')
row_index = 4
# print the row's text
print(df.iloc[row_index]['text'])
# print the row's tokenized text
print(tokenize(df.iloc[row_index]['text']))

# # load vocab
# with open('../data/vocab10k.pkl', 'rb') as f:
#     vocab = pickle.load(f)

# print(len(vocab))
# try:
#     print(vocab.get_stoi()['<pad>'])
#     # print(vocab.get_stoi()['<unk>'])
#     # print(vocab.get_stoi()['the'])
# except KeyError:
#     print("KeyError")

