import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=256, ntm=False):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.linear = nn.Linear(self.max_length * self.embedding_dim, self.hidden_size)
        self.conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=1)

    def forward(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        return x.transpose(1, 2)  # return n x len x hidden
