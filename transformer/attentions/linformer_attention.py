# Linformer: Self-Attention with Linear Complexity
# (https://arxiv.org/abs/2006.04768)
import torch
import torch.nn as nn
import math


class LinformerAttention(nn.Module):
    """
    compute Linformer attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, max_seq_length, k_proj_dim):
        super(LinformerAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # Projection from max_seq_length to k_proj_dim,
        # which is the projection dimension
        self.E = nn.Linear(max_seq_length, k_proj_dim)  # E matrix is for Key
        self.F = nn.Linear(max_seq_length, k_proj_dim)  # F matrix is for Value
        print(f"LinformerAttention initialized with k_proj_dim: {k_proj_dim}")

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        v_t = v.transpose(2, 3)
        # 1. transform Key and Value using E, F matrix
        k_proj = self.E(k_t)  # [batch_size, head, d_tensor, k_proj_dim]
        v_proj = self.F(v_t)  # [batch_size, head, d_tensor, k_proj_dim]

        # 2. dot product Query with Key^T to compute similarity

        # scaled dot product
        # score : [batch_size, head, length, k_proj_dim]
        score = (q @ k_proj) / math.sqrt(d_tensor)

        # 3. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 4. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 5. multiply with Value and change k_proj_dim back to length
        # v_proj needs to be transposed back to [batch_size, head, k_proj_dim, d_tensor]
        # [batch_size, head, length, d_tensor]
        result = score @ v_proj.transpose(2, 3)
        return result
