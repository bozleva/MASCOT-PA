import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim
from toolkits.encoders.BRAN import BRANEncoder, BRANEncoderLayer

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=256):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.BRANEncoder = BRANEncoder(BRANEncoderLayer(d_model=self.embedding_dim, nhead=10, padding=1), num_layers=2)
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs):
        inputs = self.BRANEncoder(inputs)
        inputs = self.cnn(inputs)
        return inputs

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size
