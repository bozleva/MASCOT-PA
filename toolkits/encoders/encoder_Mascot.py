import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkits.encoders.Mascot import MascotEncoder, MascotEncoderLayer

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=256):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.Mascot = MascotEncoder(MascotEncoderLayer(d_model=self.embedding_dim, nhead=10), num_layers=2)
        self.drop = nn.Dropout(0.1)

    def forward(self, inputs):
        inputs = self.Mascot(inputs)
        inputs = self.drop(inputs)
        inputs = inputs * (torch.tanh(F.softplus(inputs)))
        return inputs
