import numpy as np
import torch
import torch.nn as nn

class LEAM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes=1, word_embedding=None, label_embedding=None, phrase_window=5):
        super().__init__()

        if word_embedding is not None and label_embedding is not None:
            assert word_embedding.shape[1] == label_embedding.shape[1]
            assert torch.is_tensor(word_embedding) and torch.is_tensor(label_embedding)

            self.V = nn.Embedding(vocab_size, embed_dim, _weight=word_embedding, padding_idx=0)
            self.C = nn.Parameter(label_embedding)
        else:
            self.V = nn.Embedding(vocab_size, embed_dim)
            self.C = nn.Parameter(torch.normal(0, 1, size=(n_classes, embed_dim)))

        self.P = embed_dim
        self.K = n_classes
        self.R = phrase_window

        # self.f1_w = torch.nn.Parameter(torch.normal(0, 1, size=(self.R*2+1, )), requires_grad=True)
        # self.f1_b = torch.nn.Parameter(torch.normal(0, 1, size=(self.K, )))
        self.f1 = nn.Conv2d(1, 1, (1, 2*self.R+1), bias=True)
        self.f2 = nn.Linear(self.P, self.K)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.5)

    def _embed_sentence(self, x):
        x = self.V(x)
        return x

    def _label_attention(self, x):
        x_hat = torch.divide(x, torch.norm(x, dim=2, keepdim=True)+1e-3)
        l_hat = torch.divide(self.C, torch.norm(self.C, dim=1, keepdim=True)+1e-3)

        G_hat = torch.matmul(l_hat.unsqueeze(dim=0), x_hat.permute(0, 2, 1))
        return G_hat

    def forward(self, x):
        x = self._embed_sentence(x)
        G_hat = self._label_attention(x)

        pad_number = self.R
        G_hat = torch.nn.functional.pad(G_hat, (pad_number, pad_number))

        # there is a better way to implement label-phrase consistency
        # by reshaping G_hat, that is left for future work
        # m = []
        # for i in range(self.R, self.R + x.shape[1]):
        #     G_sub = G_hat[..., i-self.R:i+self.R+1]
        #     u_l = self.relu(torch.matmul(G_sub, self.f1_w) + self.f1_b)
        #     m.append(u_l)
        #
        # m = torch.stack(m, dim=-1)

        m = self.f1(G_hat.unsqueeze(dim=1)).squeeze(dim=1)
        m = self.relu(m)

        m = m.max(dim=1)[0]

        beta = torch.softmax(m, dim=1).reshape(-1, x.shape[1], 1)
        x = torch.mean(beta * x, dim=1)
        x = self.dropout(x)
        x = self.f2(x)
        return x