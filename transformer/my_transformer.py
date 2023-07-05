import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding


class MyTransformer(nn.Module):
    def __init__(self, Config, vocab_size, output_dim, device):
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
        if hasattr(Config, 'ATTENTION_TYPE'):
            attention_type = Config.ATTENTION_TYPE
        else:
            attention_type = 'dot_product'  # default scale dot product attention
        if hasattr(Config, 'POSITIONAL_ENCODING'):
            self.use_pe = Config.POSITIONAL_ENCODING
        else:
            self.use_pe = True
        self.vocab_size = vocab_size
        self.d_model = Config.D_MODEL
        self.output_dim = output_dim
        self.n_head = Config.N_HEAD
        self.drop_prob = Config.DROPOUT
        self.max_len = Config.MAX_SEQ_LENGTH
        self.ffn_hidden = Config.FFN_HIDDEN
        self.n_layers = Config.NUM_LAYERS

        if Config.USE_GLOVE:
            # Use pretrained GloVe embeddings
            import torchtext
            glove = torchtext.vocab.GloVe(name='6B',
                                          dim=Config.GLOVE_EMBEDDING_SIZE,
                                          cache=Config.GLOVE_CACHE_DIR)
            self.embedding = torch.nn.Embedding.from_pretrained(
                glove.vectors, freeze=True)
            print(f"Loaded GloVe embeddings of shape: {glove.vectors.shape}")
        else:
            self.embedding = nn.Embedding(
                vocab_size, self.d_model, padding_idx=0)
        if self.use_pe:
            self.positional_encoding = PositionalEncoding(
                self.d_model, self.max_len, device)
        self.drop_out = nn.Dropout(p=self.drop_prob)
        # Support multiple layers
        self.layers = nn.ModuleList(
            [EncoderLayer(self.d_model, self.ffn_hidden, self.n_head,
                          self.drop_prob, self.max_len, attention_type)
             for _ in range(self.n_layers)])
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
