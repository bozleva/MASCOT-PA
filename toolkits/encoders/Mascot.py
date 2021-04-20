import copy
from typing import Optional, Any
import os
import shutil

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


class MascotEncoder(Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MascotEncoder, self).__init__()
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


class Mish(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class MascotEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_length=128):
        super(MascotEncoderLayer, self).__init__()
        self.mh_self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Mascot
        self.m_conv_0 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=1)
        self.m_conv_active_0 = Mish()
        self.m_conv_1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=1)
        self.m_bn = nn.BatchNorm1d(d_model)
        self.m_ln = LayerNorm(d_model)
        self.m_conv_active_1 = Mish()
        self.m_no_active_conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1, padding=0)

        # FFN
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.ffn_active = Mish()


    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2 = self.mh_self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src3 = self.m_conv_active_0(self.m_conv_0(src.transpose(1, 2)))
        src3 = self.m_conv_active_1(self.m_conv_1(src3))
        src3 = self.m_no_active_conv_2(src.transpose(1, 2) + self.dropout(src3))

        src = src + self.dropout(src2) + self.dropout(src3.transpose(1, 2))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.ffn_active(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
