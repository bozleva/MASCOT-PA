import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=256):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=10), num_layers=2)
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        inputs = self.transformer_encoder(inputs)
        return inputs
