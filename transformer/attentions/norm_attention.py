# Adopt from The Devil in Linear Transformer 
# and Scaling TransNormer to 175 Billion Parameters
# (https://arxiv.org/abs/2210.10340)
# (https://arxiv.org/abs/2307.14995)
import torch
import torch.nn as nn
import math

from ..layer_norm import LayerNorm


class NormAttention(nn.Module):
    """
    compute norm attention from researches by Shanghai AI Laboratory

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, d_tensor, normalization="layer-norm"):
        super(NormAttention, self).__init__()
        assert normalization in ["layer-norm", "srms", "rms"]
        self.normalization = normalization
        if normalization == "layer-norm":
            self.layer_norm = LayerNorm(d_tensor)
        print(f"NormAttention with {normalization}")

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        # transpose [batch_size, head, d_tensor, length]
        k_t = k.transpose(2, 3)
        # scaled dot product
        # score : [batch_size, head, length, length]
        score = (q @ k_t)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. multiply with Value
        result = score @ v  # [batch_size, head, length, d_tensor]

        # 4. apply normalization
        if self.normalization == "layer-norm":
            result = self.layer_norm(result)
        # srms formula: x / ( ||x||2 / sqrt(d_tensor) ), l2 norm is applied to the length dimension
        elif self.normalization == "srms":
            result = result / (torch.norm(result, dim=2, keepdim=True) / math.sqrt(d_tensor))
        elif self.normalization == "rms":
            raise NotImplementedError("RMS normalization is not implemented yet.")

        return result
