# -*- coding: utf-8 -*-
# @Time    : 6/17/22 7:45 PM
# @Author  : LIANYONGXING
# @FileName: model.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=200, num_classes=2, filter_size=(2,3,4), num_filters=256):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_size]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_size), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x

    def forward(self, x):
        emb_x = self.embedding(x)
        emb_x = emb_x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(emb_x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":

    model = TextCNN(vocab_size=1000)
    inputs = torch.tensor(np.random.random_integers(1, 30, (1, 30)))
    logits = model(inputs)
    y = torch.tensor([1])
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(loss)