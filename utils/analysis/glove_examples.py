import torch
import torch.nn as nn
import torchtext
import pickle

glove = torchtext.vocab.GloVe(name='6B', dim=300, cache='/vol/bitbucket/fh422/torchtext_cache')
print("Loaded glove {} words".format(len(glove)))

with open('../data/vocab300k.pkl', 'rb') as f:
    vocab = pickle.load(f)
print("Loaded vocab {} words".format(len(vocab)))

temp = ['you', 'are', 'an', 'idiot']
# get the indices of the words
vocab_indices = [vocab.get_stoi()[word] for word in temp]
glove_indices = [glove.get_stoi()[word] for word in temp]
print("Vocab indices: {}".format(vocab_indices))
print("Glove indices: {}".format(glove_indices))
