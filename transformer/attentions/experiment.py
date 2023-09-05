# Reference: https://github.com/hyunwoongko/transformer
import torch
import torch.nn as nn
import math


class Experiment(nn.Module):
    """
    compute DiagAttention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, max_seq_length, block_size=15):
        super(Experiment, self).__init__()
        self.block_size = block_size
        self.softmax = nn.Softmax(dim=-1)
        # self.Wp = nn.Linear(
        #         max_seq_length, max_seq_length, bias=False).weight
        print(f"Using Experiment with block size {block_size}")

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        
        # in diag attention, we divide q and k into length / w blocks,
        # where w is the block size
        # we only compute the attention within each block
        # for ex, if length = 16, w = 4, then we have 4 blocks
        # q0 @ k0_t, q1 @ k1_t, q2 @ k2_t, q3 @ k3_t
        # and we concatenate the results to get the final QK^T
        w = self.block_size
        num_blocks = length // w
        q_blocks = torch.split(q, w, dim=2)  # list of [batch_size, head, w, d_tensor]
        k_blocks = torch.split(k_t, w, dim=3)  # list of [batch_size, head, d_tensor, w]
        # score = torch.zeros(batch_size, head, length, length).to(q.device)
        # init score to be matrix of -10000
        score = torch.ones(batch_size, head, length, length).to(q.device) * -10000

        # TODO (optional): vectorize this for loop
        for i in range(num_blocks):
            score_block = q_blocks[i] @ k_blocks[i]  # [batch_size, head, w, w]
            score[:, :, i * w: (i + 1) * w, i * w: (i + 1) * w] = score_block

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        return result
