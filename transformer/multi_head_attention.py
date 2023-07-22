# Reference: https://github.com/hyunwoongko/transformer
import torch.nn as nn

from .scale_dot_product_attention import ScaleDotProductAttention
from .additive_attention import AdditiveAttention
from .position_aware_attention_scaling import PositionAwareAttentionScaling
from .sim_attention import SimAttention
from .soft_attention import SOFTAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, max_seq_length, attention_type):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.q_same_as_k = False

        valid_a_types = ['dot_product', 'additive', 'paas',
                         'paas-linear', 'simal1', 'simal2', "soft"]
        if attention_type not in valid_a_types:
            raise ValueError(
                f"attention_type should be one of {valid_a_types}, but got {attention_type}")
        if attention_type == 'dot_product':
            self.attention = ScaleDotProductAttention()
        elif attention_type == 'additive':
            d_tensor = d_model // self.n_head
            self.attention = AdditiveAttention(d_tensor)
        elif attention_type == 'paas':
            self.attention = PositionAwareAttentionScaling(max_seq_length)
        elif attention_type == 'paas-linear':
            self.attention = PositionAwareAttentionScaling(
                max_seq_length, wp_init='linear')
        elif attention_type == 'simal1':
            self.attention = SimAttention(use_l1_norm=True)
        elif attention_type == 'simal2':
            self.attention = SimAttention(use_l1_norm=False)
        elif attention_type == 'soft':
            self.q_same_as_k = True
            self.attention = SOFTAttention()

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
