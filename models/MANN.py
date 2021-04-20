import sys
sys.path.append('..')
import torch
import numpy as np
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import toolkits.embedding as embedding


class MANN(nn.Module):

    def __init__(self, word_vec_mat, max_length, encoder, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.sentence_encoder = encoder.Encoder(max_length, hidden_size=hidden_size, word_embedding_dim=word_embedding_dim, pos_embedding_dim=pos_embedding_dim)

        # MANN
        self.mem_size = 128
        self.mann_mem = nn.Parameter(torch.rand([self.mem_size, self.hidden_size]), requires_grad=False)
        self.mem_weight = nn.Parameter(torch.rand(self.mem_size), requires_grad=False)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.pool = nn.MaxPool1d(max_length)
        self.drop = nn.Dropout(0.2)
        self.cost = nn.CrossEntropyLoss()
        self.na = args.na_rate

    def loss(self, logits, label):
        N = logits.size(-1)
        cost = self.cost(logits.view(-1, N), label.view(-1))
        return cost

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __euclid_dist__(self, S, Q):
        return self.__dist__(Q.unsqueeze(2), S.unsqueeze(1), 3)

    def mann(self, K_i, update_mann):
        W_t = F.softmax(torch.matmul(K_i.detach(), self.mann_mem.transpose(0, 1)), dim=1)
        R_t = torch.matmul(W_t, self.mann_mem)

        if update_mann:
            # find min index
            weight_list = self.mem_weight.tolist()
            sorted_weight = weight_list
            sorted_weight.sort()
            new_idx_list = [weight_list.index(num) for num in sorted_weight[:R_t.shape[0]]]

            weight_list = torch.tensor(weight_list).cuda()
            for idx, w_idx in enumerate(new_idx_list):
                # update MANN memory
                self.mann_mem[w_idx] = R_t[idx]
                # update memory weight
                weight_list[w_idx] = 0.5 * len(new_idx_list)
                weight_list = weight_list + W_t[idx]

            weight_list = F.softmax(weight_list, dim=0)
            for idx in range(len(self.mem_weight)):
                self.mem_weight[idx] = weight_list[idx]

        return R_t + K_i

    def forward(self, support, query, N, K, Q, update_mann=True):

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

        # MANN
        support = self.mann(support, update_mann)
        query = self.mann(query, update_mann)

        # Dropout
        support = self.drop(support)
        query = self.drop(query)

        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1,  N * Q + self.na * Q, self.hidden_size) # (B, total_Q, D)

        support = torch.mean(support, 2) # Calculate prototype for each class

        logits = self.__euclid_dist__(support, query)  # (B, total_Q, N)

        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred, 0