import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
import toolkits.embedding as embedding
from toolkits.encoders.Mascot import MascotEncoder, MascotEncoderLayer


class Mish(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class MainModel(nn.Module):

    def __init__(self, word_vec_mat, max_length, encoder, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100, drop=True):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.args = args
        self.B = args.batch
        self.K = args.K
        self.na = args.na_rate
        self.drop = drop
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.sentence_encoder = encoder.Encoder(max_length, hidden_size=hidden_size, word_embedding_dim=word_embedding_dim, pos_embedding_dim=pos_embedding_dim)
        self.cost = nn.CrossEntropyLoss()

        self.MLProc = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), Mish(), nn.Linear(self.hidden_size, 1))
        self.MLP_out = nn.Sequential(Mish(), nn.Linear(self.K, 1))
        self.layer_norm = nn.LayerNorm(self.word_embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.word_embedding_dim*2)

        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.Mascot = MascotEncoder(MascotEncoderLayer(d_model=self.word_embedding_dim, nhead=10), num_layers=2)

    def __Mish__(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

    def __cosine_dist__(self, S, Q):
        return torch.matmul(Q, S.transpose(1, 2))

    def loss(self, logits, label):
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).float())

    def instance_attention(self, s_input, q_input, N, K, NQ_na):
        s_shape = s_input.size()
        q_shape = q_input.size()
        support = s_input.view(-1, 1, N * K, self.max_length, s_shape[-1]).expand(-1, NQ_na, N * K, self.max_length, s_shape[-1]).reshape(-1, self.max_length, s_shape[-1])
        query = q_input.view(-1, NQ_na, 1, self.max_length, q_shape[-1]).expand(-1, NQ_na, N * K, self.max_length, q_shape[-1]).reshape(-1, self.max_length, q_shape[-1])

        att = support @ query.transpose(1, 2)
        support_ = F.softmax(att, 2) @ query
        query_ = F.softmax(att.transpose(1, 2), 2) @ support
        enhance_support = torch.cat([support, support_, torch.abs(support - support_), support * support_], -1)
        enhance_query = torch.cat([query, query_, torch.abs(query - query_), query * query_], -1)
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        support = self.__Mish__(enhance_support)
        query = self.__Mish__(enhance_query)
        paired_ = torch.cat([support, torch.zeros(len(support), 1, self.hidden_size).cuda(), query], 1)

        paired_sq = self.Mascot(paired_)
        paired_sq = torch.cat([torch.max(paired_sq, -2)[0], torch.mean(paired_sq, -2)], -1)
        paired_sq = self.__Mish__(paired_sq)

        return self.layer_norm2(paired_sq)


    def forward(self, support, query, N, K, Q):

        NQ_na = N * Q + self.na * Q
        # Embedding input words
        support_embedding = self.embedding(support)
        query_embedding = self.embedding(query)

        # Encode sentence
        support = self.sentence_encoder(support_embedding)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query_embedding)  # (B * total_Q, D)

        batch = support.size(0)//(N*K)

        # Normalize input sentence representation
        support = self.layer_norm(support)
        query = self.layer_norm(query)
        
        # Instance Attention module
        paired_sq = self.instance_attention(support, query, N, K, NQ_na)
        paired_sq = paired_sq.view(self.B, NQ_na, N, K, -1) # (B, N, K, D)
        logits = self.MLP_out(self.MLProc(self.dropout(paired_sq)).squeeze(-1))
        logits = logits.view(batch, NQ_na, N)

        if self.na > 0:
            minn, _ = logits.min(-1)
            logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
            _, pred = torch.max(logits.view(-1, N + 1), 1)
        else:
            _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred, 0
