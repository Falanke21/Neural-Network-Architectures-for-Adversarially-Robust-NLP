import torch
from torch.utils.data import Dataset

from utils.tokenizer import MyTokenizer

class YelpReviewDataset(Dataset):
    def __init__(self, df, vocab, max_seq_length):
        self.df = df
        self.vocab = vocab
        self.seq_length = max_seq_length
        self.tokenizer = MyTokenizer(
            vocab, max_seq_length, remove_stopwords=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']  # text is a string
        # get indices of tokens from the text
        indices = torch.tensor(self.tokenizer(text), dtype=torch.long)
        label = self.df.loc[idx, 'label']
        # also return text for adversarial training
        return (indices, label, text)
