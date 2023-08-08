# Gated Linear Unit described in FLASH
# https://arxiv.org/abs/2202.10447
import torch.nn as nn


class GatedLinearUnit(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(GatedLinearUnit, self).__init__()
        self.Wu = nn.Linear(d_model, hidden)
        self.Wv = nn.Linear(d_model, hidden)
        self.Wo = nn.Linear(hidden, d_model)

        self.activation_func_u = nn.ReLU()
        self.activation_func_v = nn.ReLU()
        self.dropout_u = nn.Dropout(p=drop_prob)
        self.dropout_v = nn.Dropout(p=drop_prob)
        print('Using Gated Linear Unit')

    def forward(self, x):
        U = self.Wu(x)
        U = self.activation_func_u(U)
        U = self.dropout_u(U)
        V = self.Wv(x)
        V = self.activation_func_v(V)
        V = self.dropout_v(V)

        # element-wise multiplication from U and V
        x = U * V
        x = self.Wo(x)
        return x
