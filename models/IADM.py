import sys
sys.path.append('..')

import numpy as np
import json
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import toolkits.embedding as embedding


class IADM(nn.Module):

    def __init__(self, word_vec_mat, max_length, encoder, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.sentence_encoder = encoder.Encoder(max_length, hidden_size=hidden_size, word_embedding_dim=word_embedding_dim, pos_embedding_dim=pos_embedding_dim)
        self.pool = nn.MaxPool1d(max_length)
        self.drop = nn.Dropout(0.2)
        self.cost = nn.CrossEntropyLoss()
        self.na = args.na_rate
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.g_func = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1), nn.Sigmoid())


    def loss(self, logits, label):
        N = logits.size(-1)
        cost = self.cost(logits.view(-1, N), label.view(-1))
        return cost

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def __g_func__(self, x, beta=1e-6):
        x = self.g_func(x)
        return x + beta

    def __iadm_dist__(self, S, Q, beta=1e-5):
        norm_S = S / torch.pow(torch.pow(S, 2).sum(-1).unsqueeze(-1), 0.5)
        norm_Q = Q / torch.pow(torch.pow(Q, 2).sum(-1).unsqueeze(-1), 0.5)
        np_S = norm_S / self.__g_func__(S)
        np_Q = norm_Q / self.__g_func__(Q)
        return - torch.pow(np_Q.unsqueeze(2) - np_S.unsqueeze(1), 2).sum(-1) + beta

    def forward(self, support, query, N, K, Q):

        # Embedding input words
        support_embedding = self.embedding(support)
        query_embedding = self.embedding(query)

        # Encode sentence
        support = self.sentence_encoder(support_embedding) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query_embedding) # (B * total_Q, D)

        support = self.pool(support.transpose(1, 2)).squeeze(2)
        query = self.pool(query.transpose(1, 2)).squeeze(2)

        support = self.layer_norm(support)
        query = self.layer_norm(query)

        hidden_size = support.size(-1)

        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, N * Q + self.na * Q, hidden_size) # (B, total_Q, D)

        support = torch.mean(support, 2)  # Calculate prototype for each class

        logits = self.__iadm_dist__(support, query)  # (B, total_Q, N)

        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred, 0