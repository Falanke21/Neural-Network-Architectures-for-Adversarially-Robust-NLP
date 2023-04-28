import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from encoder import EncoderLayer


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, output_dim, n_head, drop_prob, device):
        super(MyTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_head = n_head
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            d_model, 512, device)
        self.encoder = EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x is a batched one-hot vector (batch_size, seq_len, vocab_size)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.positional_encoding(x)
        x = self.encoder(x, None)

        # taking the mean across the sequence dimension
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, output_dim)
        return x
