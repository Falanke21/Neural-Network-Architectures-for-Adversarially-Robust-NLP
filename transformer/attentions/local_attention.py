# Reference: https://github.com/hyunwoongko/transformer
import torch
import torch.nn as nn
import math


class LocalAttention(nn.Module):
    """
    compute LocalAttention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, r=10):
        super(LocalAttention, self).__init__()
        self.r = r
        self.softmax = nn.Softmax(dim=-1)
        print("Using LocalAttention with r = {}".format(r))

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        # scaled dot product
        # score : [batch_size, head, length, length]
        score = (q @ k_t) / math.sqrt(d_tensor)

        # Calculate the neighborhood boundaries
        lower_bound = max(0, length - self.r)
        upper_bound = length

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # TODO check correctness
        # Mask the score matrix to only allow attention within the neighborhood
        score = score.masked_fill(
            torch.triu(torch.ones_like(score), diagonal=upper_bound) == 1, -10000)
        score = score.masked_fill(
            torch.tril(torch.ones_like(score), diagonal=lower_bound) == 1, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        return result
