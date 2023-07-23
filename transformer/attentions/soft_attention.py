# SOFT: Softmax-free Transformer with Linear Complexity
# (https://arxiv.org/abs/2110.11945)
import torch
import torch.nn as nn
import math


class SOFTAttention(nn.Module):
    """
    compute softmax-free attention

    Query : given sentence that we focused on (decoder)
    Key : key is identical to Query (encoder)!
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(SOFTAttention, self).__init__()
        print("Using SOFT Attention")

    def forward(self, q, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = q.size()

        # 1. add an extra dimension to q and k for broadcasting
        # q: [batch_size, head, length, d_tensor] -> [batch_size, head, length, 1, d_tensor]
        # k: [batch_size, head, length, d_tensor] -> [batch_size, head, 1, length, d_tensor]
        q_extra_dim = q.unsqueeze(3)
        k_extra_dim = q.unsqueeze(2)

        # 2. compute similarity by subtracting q and k
        difference = q_extra_dim - k_extra_dim  # [batch_size, head, length, length, d_tensor]

        # Remark: with an extra dimension, the resulting tensor is quite large,
        # resulting in CUDA memory error. To avoid this, we can use for loop in the future.

        # 3. compute the squared l2 norm of the difference
        # [batch_size, head, length, length, d_tensor] -> [batch_size, head, length, length]
        squared_l2_norm = torch.sum(difference ** 2, dim=-1)

        # scaling by -1/(2 * sqrt(d_tensor))
        # [batch_size, head, length, length] -> [batch_size, head, length, length]
        score = (-1 / (2 * math.sqrt(d_tensor))) * squared_l2_norm

        # 4. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        
        # 5. pass them to exp function as the paper proposed
        score = torch.exp(score)

        # 6. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        return result
