# Position-Aware Attention Scaling implementaion from
# Towards Robust Vision Transformer (https://arxiv.org/abs/2105.07926)
import torch
import torch.nn as nn
import math


class PositionAwareAttentionScaling(nn.Module):
    """
    compute position-aware attention scaling
    add a learnable position importance matrix Wp in NxN, 
    which presents the importance of each pair of q-k.

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, max_seq_length):
        super(PositionAwareAttentionScaling, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.Wp = nn.Parameter(torch.ones(max_seq_length, max_seq_length))  # TODO check more about this in their appendix
        print("Using Position-Aware Attention Scaling")

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t)
        score = torch.mul(score, self.Wp)  # apply position-aware attention scaling
        score = score / math.sqrt(d_tensor)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        result = score @ v

        return result
