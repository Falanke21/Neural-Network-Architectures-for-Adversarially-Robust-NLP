import torch
import torch.nn as nn

from .encoder import EncoderLayer
from .positional_encoding import PositionalEncoding


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_hidden, output_dim, n_head, drop_prob, max_len, n_layers, device):
        """
        :param vocab_size: int - The size of the vocabulary of the input sequence.
        :param d_model: int - The dimensionality of the model's hidden states and embeddings.
        :param ffn_hidden: int - The size of the feedforward layer in the encoder layers.
        :param output_dim: int - The size of the output layer.
        :param n_head: int - The number of attention heads in the multi-head attention layer.
        :param drop_prob: float - The dropout probability applied to the encoder layers.
        :param max_len: int - The maximum length of the input sequence.
        :param n_layers: int - The number of encoder layers in the model.
        :param device: str - The device (e.g. 'cpu' or 'cuda') where the model will be run.
        """
        super(MyTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_head = n_head
        self.drop_prob = drop_prob
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x is a batched one-hot vector (batch_size, seq_len, vocab_size)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.positional_encoding(x)
        x = self.drop_out(x)
        for layer in self.layers:
            x = layer(x, None)  # (batch_size, seq_len, d_model)

        # taking the mean across the sequence dimension
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, output_dim)
        return x
