# Reference: https://github.com/hyunwoongko/transformer
import torch
import torch.nn as nn
import math


class REVAttention(nn.Module):
    """
    compute ReLU Value Attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, use_relu_regularization=False, lambda_=1e-6, use_gpu=True):
        super(REVAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.use_rreg = use_relu_regularization
        self.lambda_ = lambda_
        self.use_gpu = use_gpu
        # initialize regularization sum as a double tensor
        self.reset_regularization()
        print(f"Using ReLU Value Attention")
        if self.use_rreg:
            print(f"Also using ReLU regularization with lambda {self.lambda_}")

    def reset_regularization(self):
        """
        Reset regularization sum.
        Need to be called before each forward pass.
        """
        device = torch.device(
            'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        # initialize regularization sum as a double tensor
        self.relu_regularization = torch.tensor(0., dtype=torch.double, device=device)

    def get_regularization(self):
        """
        Get regularization sum.
        Throw error if regularization is not used.
        """
        if self.use_rreg:
            return self.relu_regularization
        else:
            raise ValueError("ReLU regularization is not enabled in Config!")

    def _update_regularization(self, v: torch.Tensor):
        """
        Update regularization sum.
        The regularization sum is scaled by lambda, and it's a sum of all ReLU values.
        """
        if self.use_rreg:
            self.relu_regularization += (v.sum() * self.lambda_)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # Apply ReLU to Value
        v = torch.nn.functional.relu(v)

        # If we want to apply ReLU regularization, we need to keep track of the
        # sum of ReLU values here
        self._update_regularization(v)

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
