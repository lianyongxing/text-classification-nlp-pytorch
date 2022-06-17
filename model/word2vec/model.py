# -*- coding: utf-8 -*-
# @Time    : 6/17/22 2:34 PM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

import torch.nn as nn
import torch
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, center):
        emb_center = self.embeddings(center)
        outputs = self.w(emb_center)
        return outputs


# n v 1
class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2*context_size*embedding_dim, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, ctx):
        emb_ctx = self.embeddings(ctx).view((1, -1))
        hidd_out = F.relu(self.linear1(emb_ctx))
        output = self.linear2(hidd_out)
        return output


if __name__ == '__main__':

    # # skip-gram 架构
    # model = SkipGram(10, 16)
    # input1 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).unsqueeze(0)
    # input2 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).unsqueeze(0)
    # criterion = nn.CrossEntropyLoss()
    #
    # o = model(input1)
    # print(o.shape)
    # loss = criterion(o, input2)
    # print(loss)

    # # cbow 架构
    model = CBOW(10, 16, 2, 200)
    ctx = torch.tensor([4, 5, 1, 8]).unsqueeze(0)
    center = torch.tensor([0])
    o = model(ctx)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(o, center)
    print(loss)
