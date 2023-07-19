# Simple softmax-free attention SimA implementaion from
# SimA: Simple Softmax-free Attention for Vision Transformers
# (https://arxiv.org/abs/2206.08898)

import torch
import torch.nn as nn
import math


class SimAttention(nn.Module):
    """
    compute simple softmax-free attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, use_l1_norm=True):
        """
        :param use_l1_norm: whether to use l1 norm or l2 norm
        """
        super(SimAttention, self).__init__()
        self.use_l1_norm = use_l1_norm

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. normalize Query and Key across the token dimension
        if self.use_l1_norm:
            norm_q = torch.norm(q, p=1, dim=-1, keepdim=True)
            norm_k = torch.norm(k, p=1, dim=-1, keepdim=True)
        else:  # use l2 norm
            norm_q = torch.norm(q, p=2, dim=-1, keepdim=True)
            norm_k = torch.norm(k, p=2, dim=-1, keepdim=True)
        q = q / norm_q
        k = k / norm_k
        # 2. dot product Query with Key^T to compute similarity
        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        # scaled dot product
        # score : [batch_size, head, length, length]
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 3. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # we don't need softmax because we already normalized q and k

        # 4. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        return result
