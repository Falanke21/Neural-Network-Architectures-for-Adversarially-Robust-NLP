import os

import numpy as np
import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding


class MyTransformer(nn.Module):
    def __init__(self, Config, vocab_size, output_dim, device):
        """
        :param vocab_size: int - The size of the vocabulary of the input sequence.
        :param ffn_hidden: int - The size of the feedforward layer in the encoder layers.
        :param output_dim: int - The size of the output layer.
        :param device: str - The device (e.g. 'cpu' or 'cuda') where the model will be run.
        """
        super(MyTransformer, self).__init__()
        if hasattr(Config, 'POSITIONAL_ENCODING'):
            self.use_pe = Config.POSITIONAL_ENCODING
        else:
            self.use_pe = True
        self.vocab_size = vocab_size
        self.d_model = Config.D_MODEL
        self.output_dim = output_dim
        self.drop_prob = Config.DROPOUT
        self.max_len = Config.MAX_SEQ_LENGTH
        self.n_layers = Config.NUM_LAYERS

        # Embedding layer
        if Config.WORD_EMBEDDING == 'custom':
            self.embedding = nn.Embedding(
                vocab_size, self.d_model, padding_idx=0)
        elif Config.WORD_EMBEDDING == 'glove':
            # Use pretrained GloVe embeddings
            import torchtext
            glove = torchtext.vocab.GloVe(name='6B',
                                          dim=Config.GLOVE_EMBEDDING_SIZE,
                                          cache=Config.GLOVE_CACHE_DIR)
            print(f"Loading GloVe embeddings of shape: {glove.vectors.shape}")
            self.embedding = nn.Embedding.from_pretrained(
                glove.vectors, freeze=True)
        elif Config.WORD_EMBEDDING == 'paragramcf':
            assert Config.D_MODEL == 300, f"D_MODEL must be 300 for Paragramcf embeddings. Got {Config.D_MODEL} instead."
            word_embeddings_file = os.path.join(
                Config.PARAGRAMCF_DIR, "paragram.npy")
            paragramcf = torch.from_numpy(np.load(word_embeddings_file))
            print(f"Loading Paragram embeddings of shape: {paragramcf.shape}")
            self.embedding = nn.Embedding.from_pretrained(
                paragramcf, freeze=True)

        if self.use_pe:
            self.positional_encoding = PositionalEncoding(
                self.d_model, self.max_len, device)
        self.drop_out = nn.Dropout(p=self.drop_prob)

        # Support multiple layers
        if hasattr(Config, 'ATTENTION_TYPE') and Config.ATTENTION_TYPE == 'transnormer':
            # in transnormer, the first half layers use diag attention
            # and the second half layers use norm attention
            self.layers = nn.ModuleList()
            half_point = self.n_layers // 2
            for i in range(self.n_layers):
                if i < half_point:
                    Config.ATTENTION_TYPE = 'diag'
                else:
                    Config.ATTENTION_TYPE = 'norm'
                self.layers.append(EncoderLayer(Config))
        else:
            self.layers = nn.ModuleList(
                [EncoderLayer(Config) for _ in range(self.n_layers)])
        self.fc = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # For unbatched 1D input, we add a batch dimension of 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) > 2:
            raise ValueError(
                "Input must be a 1D or 2D tensor. Got tensor of shape: {}".format(x.shape))
        # x is a batched list of ids (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        if self.use_pe:
            x = x + self.positional_encoding(x)
        x = self.drop_out(x)
        for layer in self.layers:
            x = layer(x, None)  # (batch_size, seq_len, d_model)

        # taking the mean across the sequence dimension
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, output_dim)
        return x
