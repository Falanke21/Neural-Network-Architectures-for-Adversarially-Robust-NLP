# Reference: https://github.com/hyunwoongko/transformer
import torch
import torch.nn as nn
import math


class NREVAttention(nn.Module):
    """
    compute Negative ReLU Value Attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(NREVAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        print(f"Using Negative ReLU Value Attention")

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # Apply a negative ReLU to Value
        class NegativeReLU(nn.Module):
            """
            Negative ReLU activation function
            Same as ReLU, but only negative values and 0 are passed through
            """
            def __init__(self):
                super(NegativeReLU, self).__init__()
            
            def forward(self, x):
                return -torch.nn.functional.relu(-x)

        v = NegativeReLU()(v)

        # 1. dot product Query with Key^T to compute similarity
        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        # scaled dot product
        # score : [batch_size, head, length, length]
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        return result
