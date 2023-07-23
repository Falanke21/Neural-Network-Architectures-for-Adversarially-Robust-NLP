# Reference: https://github.com/hyunwoongko/transformer
import torch.nn as nn
from .attention_factory import get_attention_by_config


class MultiHeadAttention(nn.Module):

    def __init__(self, Config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = Config.N_HEAD
        self.q_same_as_k = False
        d_model = Config.D_MODEL
        max_seq_length = Config.MAX_SEQ_LENGTH

        self.attention, self.q_same_as_k = get_attention_by_config(Config)

        self.w_q = nn.Linear(d_model, d_model)
        if not self.q_same_as_k:
            self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, v = self.w_q(q), self.w_v(v)
        if not self.q_same_as_k:
            k = self.w_k(k)

        # 2. split tensor by number of heads
        q, v = self.split(q), self.split(v)
        if not self.q_same_as_k:
            k = self.split(k)

        # 3. do scale dot product to compute similarity
        if not self.q_same_as_k:
            out = self.attention(q, k, v, mask=mask)
        else:
            out = self.attention(q, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length,
                             self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(
            batch_size, length, d_model)
        return tensor
