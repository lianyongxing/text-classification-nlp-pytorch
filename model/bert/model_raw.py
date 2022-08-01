# -*- coding: utf-8 -*-
# @Time    : 7/29/22 3:28 PM
# @Author  : LIANYONGXING
# @FileName: model_raw.py
# @Software: PyCharm
# @Repo    : https://github.com/lianyongxing/text-classification-nlp-pytorch

import torch
import torch.nn as nn
import numpy
import sys
sys.path.append('../')

from transformer.model_raw import get_attn_pad_mask, MultiheadAttention
import math

n_layers = 6
d_model = 768
n_classes = 2
vocab_size = 1000
d_ff = 3072
maxlen = 10
n_segments = 2


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):

    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(maxlen, d_model)
        self.seg_embedding = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embedding(x) + self.pos_embedding(pos), + self.seg_embedding(seg)
        return self.norm(embedding)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiheadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_self_attn_mask, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh()
        )

        self.classifier = nn.Linear(d_model, n_classes)
        self.linear = nn.Linear(d_model, d_model)
        self.activate2 = gelu

        embed_weight = self.embedding.tok_embedding.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)

        # Pretrain Task:
        # 1. NSP
        h_pooled = self.fc(output[:, 0])    # first token
        logits_cls = self.classifier(h_pooled)  # predict is next

        # 2. LM
        mask_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        h_masked = torch.gather(output, 1, mask_pos)
        h_masked = self.activate2(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)

        return logits_lm, logits_cls


if __name__ == '__main__':
    model = Bert()
    print(model)