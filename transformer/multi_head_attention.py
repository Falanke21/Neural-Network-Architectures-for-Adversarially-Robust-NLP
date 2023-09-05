# Reference: https://github.com/hyunwoongko/transformer
import torch
import torch.nn as nn
from .attention_factory import get_attention_by_config


class MultiHeadAttention(nn.Module):

    def __init__(self, Config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = Config.N_HEAD
        self.q_same_as_k = False
        d_model = Config.D_MODEL
        # set multi-head attention type, default to split qkv and then concat
        self.mh_type = Config.MH_TYPE if hasattr(Config, 'MH_TYPE') else 'split'
        assert self.mh_type in ['split', 'parallel'], \
            f"Multi-head attention type {self.mh_type} not supported"

        _, self.q_same_as_k = get_attention_by_config(Config)

        self.w_q = nn.Linear(d_model, d_model)
        if not self.q_same_as_k:
            self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 2 different ways to implement multi-head attention
        if self.mh_type == 'split':
            self.w_concat = nn.Linear(d_model, d_model)
            self.attention, _ = get_attention_by_config(Config)
        elif self.mh_type == 'parallel':
            print(f"Using parallel multi-head attention")
            self.mha_list = []
            for i in range(self.n_head):
                attention, _ = get_attention_by_config(Config)
                self.mha_list.append(attention)
            self.mha_list = nn.ModuleList(self.mha_list)
            self.w_concat = nn.Linear(d_model * self.n_head, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, v = self.w_q(q), self.w_v(v)
        if not self.q_same_as_k:
            k = self.w_k(k)

        if self.mh_type == 'split':
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

        elif self.mh_type == 'parallel':
            # 2. for each head, do scale dot product to compute similarity
            # with different weight matrices (parallel)
            batch_size, length, d_model = q.size()
            # initialize output tensor, with 1 extra dummy dimension for head
            out = torch.zeros(batch_size, 1, length, d_model * self.n_head).to(q.device)

            # unsqueeze to add head dimension for compatibility
            # [batch_size, length, d_model] -> [batch_size, 1, length, d_model]
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            for i in range(self.n_head):
                attention = self.mha_list[i]
                # 3. do scale dot product to compute similarity
                if not self.q_same_as_k:
                    out[:, :, :, i*d_model:(i+1)*d_model] = attention(q, k, v, mask=mask)
                else:
                    out[:, :, :, i*d_model:(i+1)*d_model] = attention(q, v, mask=mask)
            # squeeze to remove head dimension
            # [batch_size, 1, length, num_head * d_model] -> [batch_size, length, num_head * d_model]
            out = out.squeeze(1)
            # 4. concat and pass to linear layer
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
