import copy
from typing import Optional, Any

import torch
from torch import Tensor
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class BRANEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(BRANEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class BRANEncoderLayer(Module):

    def __init__(self, d_model, nhead, padding, dropout=0.1, max_length=128):
        super(BRANEncoderLayer, self).__init__()
        self.mh_self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(d_model)

        # BRAN
        self.conv_0 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=padding)
        self.conv_relu_0 = nn.ReLU()
        self.conv_1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=padding)
        self.conv_relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2 = self.mh_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = src + self.dropout(src2)
        src2 = self.norm(src2)
        src2 = self.conv_relu_0(self.conv_0(src2.transpose(1, 2)))
        src2 = self.conv_relu_1(self.conv_1(src2))
        src2 = self.conv_2(src2)
        return self.dropout(src2.transpose(1, 2)) + src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
